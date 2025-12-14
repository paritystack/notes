# The Zig Programming Language Handbook

## Table of Contents
1.  [Introduction](#introduction)
2.  [Language Basics](#language-basics)
3.  [Memory Management](#memory-management)
4.  [Data Structures](#data-structures)
5.  [Control Flow](#control-flow)
6.  [Functions and Errors](#functions-and-errors)
7.  [Structs, Unions, and Enums](#structs-unions-and-enums)
8.  [Comptime (Compile Time Metaprogramming)](#comptime)
9.  [The Standard Library (`std`)](#the-standard-library)
10. [Build System](#build-system)
11. [C Interoperability](#c-interoperability)
12. [Testing](#testing)
13. [Advanced Topics](#advanced-topics)
14. [Common Zig Patterns](#common-zig-patterns)

---

## 1. Introduction <a name="introduction"></a>

Zig is a general-purpose programming language and toolchain for maintaining robust, optimal, and reusable software. It is often described as a "modern C" – it keeps the low-level control and simple mental model of C but adds modern features to make it safer and more ergonomic.

### The Zen of Zig
1.  Communicate intent precisely.
2.  Edge cases matter.
3.  Favor reading code over writing code.
4.  Only one obvious way to do things.
5.  Runtime crashes are better than bugs.
6.  Compile errors are better than runtime crashes.
7.  Incremental improvements.
8.  Avoid local maximums.
9.  Reduce the amount one must remember.
10. Focus on code rather than style.
11. Resource allocation may fail; resource deallocation must succeed.
12. Memory is a resource; memory must be managed.
13. Together we serve the users.

### Why Zig?
*   **No hidden control flow:** There are no hidden calls to `malloc`, no destructors, no operator overloading, and no exceptions. When you read Zig code, you know exactly what it does.
*   **No hidden allocations:** Heap allocation is manual and explicit. The language does not depend on a specific memory allocator.
*   **First-class cross-compilation:** The Zig compiler (`zig cc` and `zig c++`) can compile C and C++ code for almost any target out of the box.
*   **Comptime:** A fresh take on metaprogramming. Instead of macros or complex template metaprogramming, Zig allows you to run Zig code at compile time.

---

## 2. Language Basics <a name="language-basics"></a>

### Comments
```zig
// Single line comment

/// Doc comment (for generating documentation)
/// These attach to the next declaration.

//! Top-level doc comment (for the file/module itself)
```

### Primitive Types
Zig provides a wide variety of primitive types, including arbitrary bit-width integers.

```zig
const std = @import("std");

pub fn main() void {
    // Integers
    const a: i32 = -100;
    const b: u32 = 100;
    const c: i8 = -10;
    const d: u8 = 255;
    
    // Arbitrary bit-width integers
    const w: u7 = 127;   // Max value for 7 bits
    const x: i4 = -8;    // Min value for 4 bits signed
    
    // Floating point
    const f1: f32 = 1.0;
    const f2: f64 = 3.14159;
    const f3: f16 = 0.5;
    const f4: f128 = 1.0e100;

    // Boolean
    const truth: bool = true;
    const lie: bool = false;

    // Void (size 0)
    const v: void = {};
}
```

### Variables
Variables are declared with `const` (immutable) or `var` (mutable).

```zig
test "variables" {
    const x: i32 = 5;
    // x = 6; // Compile Error!

    var y: i32 = 5;
    y = 6; // OK

    // Type inference
    const z = 10; // Inferred as comptime_int or closest fitting type
    var name = "Zig"; // Inferred as *const [3:0]u8
}
```

### Arrays and Slices
Arrays are fixed-size. Slices are a pointer and a length.

```zig
test "arrays and slices" {
    // Array: [N]T
    var arr = [5]i32{ 1, 2, 3, 4, 5 };
    const len = arr.len; // 5

    // Inferred length
    const arr2 = [_]u8{ 'h', 'e', 'l', 'l', 'o' };

    // Slice: []T
    // A slice is a "view" into an array or memory region.
    var slice: []i32 = arr[1..4]; // Elements {2, 3, 4}
    slice[0] = 99; // Modifies arr[1] to 99

    // Slice literals
    const s2 = "Hello"; // *const [5:0]u8 (pointer to null-terminated array)
    const s3: []const u8 = "World"; // Slice of bytes
}
```

### Pointers
Zig has several pointer types to provide safety.

1.  **Single Item Pointer (`*T`)**: Points to exactly one item. No pointer arithmetic.
2.  **Many Item Pointer (`[*]T`)**: Like C pointers. Supports arithmetic. Unsafe.
3.  **Slice (`[]T`)**: Pointer + Length. Safe.
4.  **C Pointer (`[*c]T`)**: For C interop.

```zig
test "pointers" {
    var x: i32 = 10;
    const ptr: *i32 = &x; // Single item pointer
    ptr.* = 20; // Dereference syntax

    var arr = [_]i32{ 1, 2, 3 };
    const many_ptr: [*]i32 = &arr; // Decays to many-item pointer
    const second = many_ptr[1]; // Access like array
}
```

---

## 3. Memory Management <a name="memory-management"></a>

Zig requires manual memory management. There is no `malloc` in the language syntax; instead, the standard library provides the `Allocator` interface.

### The `Allocator` Interface
Most functions that allocate memory accept an `allocator` parameter.

```zig
const std = @import("std");

pub fn main() !void {
    // 1. Choose an allocator
    // GeneralPurposeAllocator is a safe allocator for debug/release
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit(); // Check for leaks
    const allocator = gpa.allocator();

    // 2. Allocate single item
    const ptr = try allocator.create(i32);
    defer allocator.destroy(ptr); // Free
    ptr.* = 123;

    // 3. Allocate slice
    const slice = try allocator.alloc(u8, 100);
    defer allocator.free(slice); // Free
    
    // Initialize slice
    @memset(slice, 0);
}
```

### ArenaAllocator
The Arena strategy allows you to free all allocated memory at once.

```zig
test "arena allocator" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit(); // Frees everything allocated in this arena
    const allocator = arena.allocator();

    const p1 = try allocator.create(i32);
    const p2 = try allocator.create(i32);
    const p3 = try allocator.alloc(u8, 50);
    
    // No need to call destroy/free for p1, p2, p3 individually.
}
```

### FixedBufferAllocator
Allocates from a fixed-size buffer (stack or pre-allocated heap). Very fast, no heap fragmentation.

```zig
test "fixed buffer allocator" {
    var buffer: [1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buffer);
    const allocator = fba.allocator();

    const x = try allocator.create(i32);
    x.* = 42;
    // Fails if buffer runs out of space
}
```

---

## 4. Data Structures <a name="data-structures"></a>

### `std.ArrayList`
A contiguous, growable array.

```zig
const std = @import("std");

test "ArrayList" {
    const allocator = std.testing.allocator;
    
    var list = std.ArrayList(i32).init(allocator);
    defer list.deinit();

    try list.append(1);
    try list.append(2);
    try list.appendSlice(&[_]i32{3, 4, 5});

    try std.testing.expectEqual(list.items.len, 5);
    try std.testing.expectEqual(list.items[0], 1);
    
    const popped = list.pop(); // 5
}
```

### `std.AutoHashMap`
A hash map where keys are automatically hashed.

```zig
test "HashMap" {
    const allocator = std.testing.allocator;

    var map = std.AutoHashMap(i32, []const u8).init(allocator);
    defer map.deinit();

    try map.put(1, "one");
    try map.put(2, "two");

    const val = map.get(1); // ?[]const u8 (Optional)
    try std.testing.expectEqualStrings(val.?, "one");

    const removed = map.remove(2); // true if existed
}
```

### `std.StringHashMap`
Optimized for string keys.

```zig
test "StringHashMap" {
    const allocator = std.testing.allocator;
    var map = std.StringHashMap(u32).init(allocator);
    defer map.deinit();

    try map.put("Alice", 100);
}
```

### Implementing a Linked List
Shows how to build custom structures with manual memory.

```zig
const std = @import("std");

fn LinkedList(comptime T: type) type {
    return struct {
        const Self = @This();
        
        pub const Node = struct {
            data: T,
            next: ?*Node,
        };
        
        head: ?*Node = null,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator) Self {
            return Self{ .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            var current = self.head;
            while (current) |node| {
                const next_node = node.next;
                self.allocator.destroy(node);
                current = next_node;
            }
        }

        pub fn append(self: *Self, data: T) !void {
            const new_node = try self.allocator.create(Node);
            new_node.* = .{ .data = data, .next = null };

            if (self.head) |head| {
                var current = head;
                while (current.next) |next| {
                    current = next;
                }
                current.next = new_node;
            } else {
                self.head = new_node;
            }
        }
    };
}

test "linked list" {
    var list = LinkedList(i32).init(std.testing.allocator);
    defer list.deinit();
    
    try list.append(10);
    try list.append(20);
}
```

---

## 5. Control Flow <a name="control-flow"></a>

### If Expression
`if` is an expression, meaning it returns a value.

```zig
fn max(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}
```

### While Loop
Supports `continue` expression (runs after each iteration).

```zig
test "while" {
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        if (i == 5) continue;
        // ...
    }
}
```

### For Loop
Iterates over slices/arrays/ranges.

```zig
test "for" {
    const items = [_]i32{1, 2, 3};
    
    // Value capture
    for (items) |item| {
        _ = item;
    }

    // Index capture
    for (items, 0..) |item, index| {
        _ = item;
        _ = index;
    }
    
    // Range (0 to 9)
    for (0..10) |i| {
        _ = i;
    }
    
    // Multi-object iteration (zip)
    const a = [_]u8{1, 2};
    const b = [_]u8{3, 4};
    for (a, b) |x, y| {
        _ = x; _ = y;
    }
}
```

### Switch
Must be exhaustive.

```zig
fn handle(x: i32) void {
    switch (x) {
        0 => std.debug.print("Zero\n", .{}),
        1, 2, 3 => std.debug.print("Small\n", .{}),
        4...10 => std.debug.print("Medium\n", .{}), // Range
        else => std.debug.print("Other\n", .{}),
    }
}
```

### Labelled Blocks & Loops
You can label loops to break/continue from nested contexts.

```zig
test "labels" {
    outer: while (true) {
        while (true) {
            break :outer;
        }
    }
    
    // Block returning value
    const x = blk: {
        const y = 1;
        break :blk y + 1;
    };
}
```

---

## 6. Functions and Errors <a name="functions-and-errors"></a>

### Functions
Parameters are immutable by default.

```zig
fn add(a: i32, b: i32) i32 {
    return a + b;
}

// Pass by pointer to mutate
fn increment(val: *i32) void {
    val.* += 1;
}
```

### Error Sets
Errors are values, defined like enums.

```zig
const FileError = error{
    NotFound,
    AccessDenied,
    DiskFull,
};
```

### Error Unions (`!T`)
A type `!i32` means "either an `i32` or an error".

```zig
fn openFile(name: []const u8) !void {
    if (name.len == 0) return FileError.NotFound;
}

// Inferred error set (common in apps)
fn doSomething() !i32 {
    return 5;
}
```

### Try, Catch, and If Error

```zig
test "error handling" {
    // 1. try: Unwraps value or returns error from current function
    // const x = try functionThatMightFail();

    // 2. catch: Handles the error
    const y = functionThatMightFail() catch |err| {
        std.debug.print("Failed: {}\n", .{err});
        return; // or return a fallback value
    };
    
    // 3. catch with value
    const z = functionThatMightFail() catch 0;
    
    // 4. if capture
    if (functionThatMightFail()) |val| {
        // Success
    } else |err| {
        // Error
    }
}

fn functionThatMightFail() !i32 {
    return FileError.AccessDenied;
}
```

### Errdefer
Execute code *only* if the function returns an error. Useful for cleaning up resources on failure paths.

```zig
fn complexOperation(allocator: std.mem.Allocator) !void {
    const p1 = try allocator.create(i32);
    
    // If subsequent steps fail, free p1.
    // If success, we return ownership of p1? (Context dependent)
    // Or if we just use it locally:
    errdefer allocator.destroy(p1);
    
    const p2 = try allocator.create(i32); // If this fails, p1 is freed.
    // ...
}
```

---

## 7. Structs, Unions, and Enums <a name="structs-unions-and-enums"></a>

### Structs
Structs are namespaced containers for fields and functions.

```zig
const Point = struct {
    x: f32,
    y: f32,
    
    // Default values
    z: f32 = 0.0,

    // Namespace function (static method)
    pub fn init(x: f32, y: f32) Point {
        return Point{ .x = x, .y = y };
    }

    // Method
    pub fn distance(self: Point) f32 {
        return @sqrt(self.x * self.x + self.y * self.y);
    }
};
```

### Packed Structs
Guaranteed memory layout, useful for hardware interaction.

```zig
const Register = packed struct {
    enable: bool,
    mode: u3,
    reserved: u4,
};
```

### Enums
Can have methods and specific integer tag types.

```zig
const Color = enum(u8) {
    Red = 1,
    Green,
    Blue,

    pub fn isRed(self: Color) bool {
        return self == .Red;
    }
};
```

### Unions
Store one value at a time.

```zig
// Tagged Union (Enum + Union)
const Value = union(enum) {
    Int: i32,
    Float: f64,
    String: []const u8,
};

fn printValue(v: Value) void {
    switch (v) {
        .Int => |i| std.debug.print("Int: {d}\n", .{i}),
        .Float => |f| std.debug.print("Float: {d}\n", .{f}),
        .String => |s| std.debug.print("String: {s}\n", .{s}),
    }
}
```

---

## 8. Comptime <a name="comptime"></a>

Zig uses `comptime` to run code during compilation. This replaces macros, C++ templates, and conditional compilation.

### Generic Functions
Types are first-class values at compile time.

```zig
fn List(comptime T: type) type {
    return struct {
        items: []T,
        len: usize,
    };
}

test "generics" {
    const IntList = List(i32);
    var list: IntList = undefined;
    _ = list;
}
```

### Compile-Time Reflection
`@typeInfo` allows inspecting types.

```zig
fn printFields(comptime T: type) void {
    const info = @typeInfo(T);
    switch (info) {
        .Struct => |s| {
            inline for (s.fields) |field| {
                std.debug.print("Field: {s}, Type: {s}\n", .{field.name, @typeName(field.type)});
            }
        },
        else => @compileError("Expected struct"),
    }
}
```

### Inline For
Unrolls loops at compile time.

```zig
test "inline for" {
    const types = .{ i32, f64, bool };
    inline for (types) |T| {
        // T is known at compile time here
        const size = @sizeOf(T);
        _ = size;
    }
}
```

---

## 9. The Standard Library <a name="the-standard-library"></a>

### `std.fs`: File System
Detailed file operations.

```zig
const std = @import("std");

pub fn fsExample() !void {
    const cwd = std.fs.cwd();

    // Write file
    try cwd.writeFile(.{ .path = "test.txt", .data = "Hello Zig" });

    // Open file
    const file = try cwd.openFile("test.txt", .{ .mode = .read_only });
    defer file.close();

    // Read entire file (allocating)
    const allocator = std.heap.page_allocator;
    const content = try file.readToEndAlloc(allocator, 1024 * 1024); // Max 1MB
    defer allocator.free(content);

    // Iterating dir
    var dir = try cwd.openDir(".", .{ .iterate = true });
    defer dir.close();
    
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        std.debug.print("{s}: {s}\n", .{entry.name, @tagName(entry.kind)});
    }
    
    // Walking dir recursively
    var walker = try dir.walk(allocator);
    defer walker.deinit();
    
    while (try walker.next()) |entry| {
        std.debug.print("Walk: {s}\n", .{entry.path});
    }
}
```

### `std.io`: Input/Output
Readers and Writers.

```zig
pub fn ioExample() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("Hello {s}\n", .{"World"});

    // Buffered Writer (for performance)
    var bw = std.io.bufferedWriter(stdout);
    const writer = bw.writer();
    try writer.print("Buffered Output\n", .{});
    try bw.flush(); // Don't forget to flush!
    
    // Custom Writer
    // Any struct with a write method can be used via std.io.Writer
}
```

### `std.process`: Process Control

```zig
pub fn processExample(allocator: std.mem.Allocator) !void {
    // Arguments
    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    
    while (args.next()) |arg| {
        std.debug.print("Arg: {s}\n", .{arg});
    }

    // Environment variables
    var env = try std.process.getEnvMap(allocator);
    defer env.deinit();
    if (env.get("HOME")) |home| {
        std.debug.print("Home: {s}\n", .{home});
    }

    // Spawning a child process
    var child = std.process.Child.init(&[_][]const u8{"ls", "-l"}, allocator);
    child.stdout_behavior = .Pipe;
    try child.spawn();
    
    // Read child output...
    _ = try child.wait();
}
```

### `std.thread`: Concurrency

```zig
fn worker(id: u32) void {
    std.debug.print("Thread {d} working\n", .{id});
}

pub fn threadExample() !void {
    // Spawn
    const t1 = try std.Thread.spawn(.{}, worker, .{1});
    const t2 = try std.Thread.spawn(.{}, worker, .{2});

    // Join
    t1.join();
    t2.join();
    
    // Mutex
    var mutex = std.Thread.Mutex{};
    mutex.lock();
    defer mutex.unlock();
}
```

### `std.net`: Networking (TCP Server)

```zig
pub fn echoServer() !void {
    const address = try std.net.Address.parseIp4("127.0.0.1", 8080);
    var server = try std.net.Address.listen(address, .{ .reuse_address = true });
    defer server.deinit();

    while (true) {
        const connection = try server.accept();
        defer connection.stream.close();

        var buffer: [1024]u8 = undefined;
        const bytes = try connection.stream.read(&buffer);
        try connection.stream.writeAll(buffer[0..bytes]);
    }
}
```

### `std.json`: JSON Handling

```zig
const User = struct {
    name: []const u8,
    age: u32,
};

pub fn jsonExample(allocator: std.mem.Allocator) !void {
    // Stringify
    const user = User{ .name = "Ziggy", .age = 10 };
    var string = std.ArrayList(u8).init(allocator);
    defer string.deinit();
    
    try std.json.stringify(user, .{}, string.writer());
    
    // Parse
    const json_src = "{\"name\": \"Ziggy\", \"age\": 10}";
    const parsed = try std.json.parseFromSlice(User, allocator, json_src, .{});
    defer parsed.deinit();
    
    std.debug.print("Parsed name: {s}\n", .{parsed.value.name});
}
```

---

## 10. Build System <a name="build-system"></a>

Zig's build system is written in Zig. It's powerful and portable.

### `build.zig` breakdown

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    // Define targets (arch, os, abi)
    const target = b.standardTargetOptions(.{});
    // Define optimization level (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall)
    const optimize = b.standardOptimizeOption(.{});

    // 1. Executable
    const exe = b.addExecutable(.{
        .name = "my-app",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Link C library if needed
    // exe.linkLibC();
    // exe.addIncludePath(b.path("include"));
    
    // Install step (copies binary to zig-out/bin)
    b.installArtifact(exe);

    // 2. Run Step
    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // 3. Test Step
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
```

### Dependency Management (`build.zig.zon`)
Since Zig 0.11+, package management is included.

```zig
// build.zig.zon
.{ 
    .name = "my-project",
    .version = "0.1.0",
    .dependencies = .{
        .zap = .{
            .url = "https://github.com/zigzap/zap/archive/refs/tags/v0.1.0.tar.gz",
            .hash = "1220...", // zig fetch --save <url> populates this
        },
    },
}
```

In `build.zig`:
```zig
const zap_dep = b.dependency("zap", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("zap", zap_dep.module("zap"));
```

---

## 11. C Interoperability <a name="c-interoperability"></a>

Zig excels at working with C.

### Importing C Headers
Use `@cImport` to parse headers and make them available in Zig.

```zig
const c = @cImport({
    @cDefine("_GNU_SOURCE", {});
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
});

pub fn main() void {
    _ = c.printf("Hello from C printf: %d\n", 42);
}
```

### Converting Types
*   `c_int` <-> `i32` (platform dependent)
*   `[*c]T` <-> C pointers
*   `[*:0]u8` <-> C strings (null terminated)

### Exporting Zig to C
Make a Zig function callable from C.

```zig
// In Zig
export fn add(a: i32, b: i32) i32 {
    return a + b;
}
```
Compile as a library: `zig build-lib main.zig -dynamic`

---

## 12. Testing <a name="testing"></a>

Zig has a built-in test runner. Tests are written alongside code.

```zig
const std = @import("std");

fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic math" {
    try std.testing.expect(add(1, 1) == 2);
}

test "expectEqual" {
    try std.testing.expectEqual(@as(i32, 42), add(40, 2));
}

test "detect memory leaks" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    try list.append(1);
    // Forgot defer list.deinit(); 
    // This test will fail reporting a memory leak!
}
```

### Fuzz Testing
Zig's standard library includes support for fuzzing.

```zig
test "fuzz example" {
    // Placeholder for fuzzing logic setup
}
```

---

## 13. Advanced Topics <a name="advanced-topics"></a>

### Async/Await (Status)
Async is a key feature of Zig (suspend/resume-based coroutines). As of recent versions (0.11/0.12), async is undergoing a major rework in the compiler ("Stage 2"). It is currently not fully available in stable releases but is a core part of the language specification. When enabled, it allows writing non-blocking code that looks synchronous (Colorblind Async).

### Vectors (SIMD)
Zig provides first-class support for SIMD vectors.

```zig
test "simd vectors" {
    const v1 = @Vector(4, f32){ 1.0, 2.0, 3.0, 4.0 };
    const v2 = @Vector(4, f32){ 5.0, 6.0, 7.0, 8.0 };
    
    // Operations happen in parallel
    const v3 = v1 + v2; // { 6.0, 8.0, 10.0, 12.0 }
    
    var arr: [4]f32 = v3;
    _ = arr;
}
```

### Inline Assembly
For bare-metal or optimization needs.

```zig
pub fn syscall(number: usize) usize {
    return asm volatile ("syscall"
        : [ret] "={{rax}}" (-> usize)
        : [number] "={{rax}}" (number)
        : "rcx", "r11"
    );
}
```

### Undefined Behavior
Zig tries to prevent UB in Debug/ReleaseSafe modes, but allows it in ReleaseFast for performance.
*   Integer overflow (panics in debug, wraps in release)
*   Out of bounds access (panics in debug, UB in ReleaseFast)
*   Reaching `unreachable` code.

---

## 14. Common Zig Patterns <a name="common-zig-patterns"></a>

### Allocator Injection
In Zig, you don't use a global allocator. instead, you pass it explicitly.

**Pattern:** Accept an allocator in `init` and store it, or accept it in the method that needs it.

```zig
const std = @import("std");

const MyString = struct {
    buffer: []u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, s: []const u8) !MyString {
        const buf = try allocator.dupe(u8, s);
        return MyString{
            .buffer = buf,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MyString) void {
        self.allocator.free(self.buffer);
    }
};
```

### Iterator Pattern
Zig iterators typically have a `next()` method that returns an optional value (`?T`).

```zig
const NumberIter = struct {
    count: i32,
    limit: i32,

    pub fn next(self: *NumberIter) ?i32 {
        if (self.count >= self.limit) return null;
        self.count += 1;
        return self.count;
    }
};

test "iterator" {
    var iter = NumberIter{ .count = 0, .limit = 3 };
    while (iter.next()) |num| {
        std.debug.print("{d}\n", .{num});
    }
}
```

### Context for Generics
Zig doesn't have closures in the traditional sense. When using generic algorithms like `std.sort`, you pass a `context` and a function that takes that context.

```zig
const std = @import("std");

pub fn main() void {
    var items = [_]i32{ 3, 1, 4, 1, 5 };
    
    // Sort descending
    // Context is {}, comparator function takes context.
    std.mem.sort(i32, &items, {}, std.sort.desc(i32));
    
    // Custom context
    const Context = struct {
        offset: i32,
        pub fn lessThan(self: @This(), a: i32, b: i32) bool {
            return (a + self.offset) < (b + self.offset);
        }
    };
    
    std.mem.sort(i32, &items, Context{ .offset = 10 }, Context.lessThan);
}
```

### Tagged Unions for State Machines
Unions are perfect for state machines where each state has different data.

```zig
const ConnectionState = union(enum) {
    Disconnected,
    Connecting: struct { retries: u8 },
    Connected: struct { socket: i32, last_ping: i64 },
    Error: []const u8,
};

fn update(state: *ConnectionState) void {
    switch (state.*) {
        .Disconnected => state.* = .Connecting{ .retries = 0 },
        .Connecting => |*data| {
            data.retries += 1;
            if (data.retries > 5) state.* = .Error{ "Timeout" };
        },
        // ...
    }
}
```

### Sentinel-Terminated Slices
Used for interacting with C APIs that expect null-terminated strings, while keeping slice safety.

```zig
test "sentinel" {
    // [:0]const u8 is a slice that ends with 0
    const s: [:0]const u8 = "hello"; 
    
    // Can access length (O(1))
    std.debug.print("Len: {d}\n", .{s.len});
    
    // Can pass to C (pointer is compatible)
    // c_function(s.ptr);
}
```

---

## Appendix: Zig Tools
*   `zig fmt`: Formats code.
*   `zig run`: Compiles and runs immediately.
*   `zig build`: Runs the build system.
*   `zig test`: Runs tests.
*   `zig translate-c`: Converts C header/code to Zig code (amazing for learning bindings).
*   `zls`: The Zig Language Server (highly recommended for editors).

```
