# Dart Programming

## Overview

Dart is a client-optimized, statically typed language developed at Google,
best known as the language behind the **Flutter** UI toolkit. It compiles
**ahead-of-time (AOT)** to native ARM/x64 for fast startup on mobile and
desktop, **just-in-time (JIT)** during development for stateful hot reload, and
to **JavaScript/WASM** for the web. Its feel sits between [TypeScript](typescript.md)
and [Kotlin](kotlin.md): C-style syntax, a strong static type system with
inference, and **sound null safety** baked into the type system.

**Key Features:**
- Sound null safety — nullable (`T?`) and non-nullable (`T`) are distinct types
- AOT to native + JIT for hot reload; compiles to JS/WASM for the web
- Single-threaded event loop with `Future`/`Stream` and `async`/`await`
- Real parallelism via **isolates** (no shared memory, message passing)
- `mixin`s and `extension`s for composition without inheritance
- `final`/`const` immutability, with compile-time `const` objects
- Everything is an object; even `int`, functions, and `null` have a type
- Batteries-included tooling: `dart` CLI, `pub` package manager, `dart format`

---

## Basic Syntax

```dart
void main() {
  // Type inference with var; the type is fixed after first assignment
  var name = 'Alice';        // String
  var age = 30;              // int

  // Explicit types
  String city = 'NYC';
  double pi = 3.14159;
  bool active = true;

  // final: set once at runtime. const: compile-time constant.
  final now = DateTime.now();
  const maxSize = 100;

  // late: non-nullable, initialized after declaration (lazily)
  late String description;
  description = 'set before first use';

  print('$name is $age, lives in $city');   // string interpolation
  print('Next year: ${age + 1}');           // expression interpolation
}
```

### Data Types

```dart
int i = 42;                  // 64-bit (arbitrary precision on web)
double d = 3.14;             // 64-bit float
num n = 7;                   // supertype of int and double
String s = 'text';           // UTF-16, single or double quotes
bool b = true;
Object o = anything;         // root of all types (non-null)
dynamic dyn = 'any';         // opts out of static checks

// Collections are generic
List<int> list = [1, 2, 3];
Set<String> set = {'a', 'b'};
Map<String, int> map = {'x': 1, 'y': 2};

// Records (Dart 3): anonymous, fixed-size, structurally typed tuples
(int, String) pair = (1, 'one');
({int x, int y}) point = (x: 3, y: 4);
print(point.x);
```

## Null Safety

A defining Dart feature: the compiler tracks nullability and refuses to let a
`null` flow into a non-nullable type. See [Type Systems](type_systems.md) for
the broader idea.

```dart
String nonNull = 'always here';   // can never be null
String? maybe = null;             // explicitly nullable

// Operators for working with nullable values
int? len = maybe?.length;         // ?.  null-aware access -> null if maybe is null
String shown = maybe ?? 'default';// ??  fallback when null
maybe ??= 'assign if null';       // ??= conditional assignment
int forced = maybe!.length;       // !   assert non-null (throws if wrong)

// Flow analysis promotes the type after a null check
void greet(String? who) {
  if (who == null) return;
  print(who.toUpperCase());       // who is promoted to String here
}
```

## Functions

```dart
// Full form
int add(int a, int b) {
  return a + b;
}

// Arrow syntax for single-expression bodies
int square(int x) => x * x;

// Optional positional params [in brackets], with defaults
String greet(String name, [String greeting = 'Hello']) => '$greeting, $name';

// Named params: {in braces}, marked `required` when mandatory
void config({required int width, int height = 100, String? label}) {}
config(width: 200, label: 'box');

// Functions are first-class; closures capture their environment
Function counter() {
  var count = 0;
  return () => ++count;     // captures `count`
}

final next = counter();
print(next());  // 1
print(next());  // 2

list.where((x) => x.isOdd).map((x) => x * 2).toList();
```

## Collections

```dart
var nums = [1, 2, 3];

// Spread, plus collection-if and collection-for inside literals
var more = [0, ...nums, if (active) 99, for (var n in nums) n * 10];

// Common operations (lazy Iterable until materialized)
nums.map((n) => n + 1);
nums.where((n) => n > 1);
nums.fold(0, (sum, n) => sum + n);
nums.firstWhere((n) => n.isEven, orElse: () => -1);

var byId = {for (var n in nums) n: 'item$n'};   // map comprehension
```

## Classes

```dart
class Point {
  final double x, y;

  // Constructor with initializing formals (assigns fields directly)
  Point(this.x, this.y);

  // Named constructor
  Point.origin() : x = 0, y = 0;

  // Factory: may return a cached/sub-typed instance
  factory Point.fromMap(Map<String, double> m) => Point(m['x']!, m['y']!);

  // Getter (computed property)
  double get magnitude => (x * x + y * y);

  @override
  String toString() => 'Point($x, $y)';
}

// Inheritance and abstract types
abstract class Shape {
  double area();                       // abstract method
  void describe() => print('area=${area()}');
}

class Circle extends Shape {
  final double r;
  Circle(this.r);
  @override
  double area() => 3.14159 * r * r;
}

// Mixins: reuse behavior across unrelated classes
mixin Loggable {
  void log(String msg) => print('[${runtimeType}] $msg');
}
class Service with Loggable {}

// Extensions: add methods to existing types without subclassing
extension StringExtras on String {
  String get reversed => split('').reversed.join();
}
print('dart'.reversed);   // trad
```

Dart 3 also adds **sealed classes** and **`switch` expressions** with
[pattern matching](pattern_matching.md) over records, lists, and objects.

## Generics

Parametric polymorphism much like [Generics](generics.md) elsewhere; type
arguments are reified (available at runtime, unlike Java's erasure).

```dart
class Box<T> {
  final T value;
  Box(this.value);
}

T firstOf<T>(List<T> items) => items.first;

// Bounded type parameters
T maxOf<T extends Comparable<T>>(T a, T b) => a.compareTo(b) >= 0 ? a : b;
```

## Asynchronous Programming

Dart runs on a single-threaded **event loop**. `Future` is a promise of a value;
`Stream` is a sequence of async events. See [async/await](async_programming.md)
and [Concurrency](concurrency.md) for the general model.

```dart
Future<String> fetchUser() async {
  await Future.delayed(Duration(seconds: 1));
  return 'Alice';
}

Future<void> main() async {
  final user = await fetchUser();        // suspends without blocking the thread
  print(user);

  // Run concurrently and join
  final results = await Future.wait([fetchUser(), fetchUser()]);

  // Streams: await each event as it arrives
  await for (final tick in countdown(3)) {
    print(tick);
  }
}

Stream<int> countdown(int from) async* {
  for (var i = from; i > 0; i--) {
    await Future.delayed(Duration(milliseconds: 500));
    yield i;                              // async generator
  }
}
```

### Isolates (parallelism)

For CPU-bound work, isolates run on separate threads with **no shared memory** —
they communicate only by passing messages, which avoids data races by design.

```dart
import 'dart:isolate';

Future<void> main() async {
  // Run a function on another isolate and get the result back
  final sum = await Isolate.run(() {
    var total = 0;
    for (var i = 0; i < 1000000000; i++) total += i;
    return total;
  });
  print(sum);
}
```

## Error Handling

Dart uses exceptions; there are no checked exceptions. See
[Error Handling](error_handling.md) for the exceptions-vs-results trade-off.

```dart
try {
  riskyOperation();
} on FormatException catch (e) {
  print('bad format: $e');        // catch a specific type
} on Exception catch (e, stack) {
  print('failed: $e\n$stack');    // value + stack trace
} catch (e) {
  rethrow;                        // re-throw to the caller
} finally {
  cleanup();                      // always runs
}

// `assert` is checked only in debug/development builds
assert(age >= 0, 'age cannot be negative');
```

## Tooling

```bash
dart create my_app        # scaffold a project
dart run                  # run main.dart
dart test                 # run tests (package:test)
dart format .             # canonical formatter (no config bikeshedding)
dart analyze              # static analysis / linter

dart pub add http         # add a dependency from pub.dev
dart pub get              # resolve dependencies (pubspec.yaml)

dart compile exe bin/main.dart   # AOT native executable

flutter run               # Flutter apps build on the same toolchain
```

## Where this connects

- [Asynchronous Programming](async_programming.md) — `Future`/`Stream` and
  `async`/`await` are Dart's event-loop concurrency, the same model as JS.
- [Concurrency](concurrency.md) — isolates give parallelism via message passing
  instead of shared-memory threads.
- [Type Systems](type_systems.md) — sound null safety and flow-based type
  promotion are the heart of Dart's static checking.
- [Generics](generics.md) — reified, bounded type parameters.
- [Pattern Matching](pattern_matching.md) — Dart 3 records, sealed classes, and
  `switch` expressions.
- [Error Handling](error_handling.md) — unchecked exceptions with typed `catch`.
- [Kotlin](kotlin.md) and [TypeScript](typescript.md) — closest siblings in
  syntax, null safety, and the "pragmatic statically typed" niche.

## Pitfalls

- **`late` is a loaded gun:** reading a `late` variable before it's assigned
  throws at runtime — you've traded a compile-time guarantee for a runtime one.
- **`!` defeats null safety:** the non-null assertion throws if you're wrong;
  prefer `?.`, `??`, or a real null check that promotes the type.
- **`const` vs `final`:** `const` must be computable at compile time and is
  canonicalized (identical `const` values are the same instance); `final` is
  just write-once at runtime. Mixing them up causes confusing errors.
- **Isolates don't share memory:** you can't pass a mutable object and mutate it
  elsewhere — everything crosses as a copy (or via `SendPort`). Large transfers
  cost serialization.
- **`dynamic` silently disables checks:** an accidental `dynamic` (e.g. from an
  untyped `Map`/JSON) turns type errors into runtime crashes — annotate or cast.
- **`Iterable` is lazy:** `map`/`where` don't run until materialized with
  `toList()`/iteration, so side effects inside them may not fire when expected.
