# Kotlin Programming

## Overview

Kotlin is a modern, statically-typed programming language that runs on the JVM and is fully interoperable with Java. Developed by JetBrains, it's the preferred language for Android development and is increasingly used for server-side applications.

**Key Features:**
- Concise syntax with less boilerplate
- Null safety built into the type system
- Interoperable with Java (100% compatible)
- Coroutines for async programming
- Extension functions
- Data classes
- Smart casts
- Functional programming support
- Excellent tooling support

---

## Basic Syntax

### Variables and Data Types

```kotlin
// Immutable variable (read-only)
val name = "Alice"
val age = 30
// name = "Bob"  // Error! Cannot reassign

// Mutable variable
var count = 0
count = 1  // OK

// Type annotations (optional due to type inference)
val name: String = "Alice"
val age: Int = 30
val pi: Double = 3.14159
val isActive: Boolean = true
val initial: Char = 'A'

// Numeric types
val byte: Byte = 127
val short: Short = 32767
val int: Int = 2147483647
val long: Long = 9223372036854775807L
val float: Float = 3.14f
val double: Double = 3.14159

// Type conversion (explicit)
val x: Int = 10
val y: Long = x.toLong()
val z: Double = x.toDouble()
val str: String = x.toString()

// Constants (compile-time)
const val MAX_SIZE = 100
const val API_KEY = "your-api-key"

// Late initialization
lateinit var user: User
// user must be initialized before use

// Lazy initialization
val heavyObject: ExpensiveObject by lazy {
    ExpensiveObject()  // Initialized on first access
}
```

### Nullable Types

Kotlin's type system distinguishes between nullable and non-nullable types.

```kotlin
// Non-nullable (default)
var name: String = "Alice"
// name = null  // Compilation error!

// Nullable
var nullableName: String? = "Bob"
nullableName = null  // OK

// Safe call operator (?.)
val length = nullableName?.length  // Returns Int? (null if nullableName is null)

// Elvis operator (?:) - provide default value
val len = nullableName?.length ?: 0  // Returns 0 if null

// Not-null assertion (!!)
val len2 = nullableName!!.length  // Throws NPE if null (use sparingly)

// Safe cast (as?)
val obj: Any = "Hello"
val str: String? = obj as? String  // null if cast fails
val num: Int? = obj as? Int  // null

// Let function (execute block if not null)
nullableName?.let {
    println("Name is $it")
    println("Length is ${it.length}")
}

// Check for null
if (nullableName != null) {
    // Smart cast: nullableName is now String (non-nullable)
    println(nullableName.length)
}
```

### String Operations

```kotlin
// String templates
val name = "Alice"
val age = 30
val message = "My name is $name and I am $age years old"
val calculation = "Sum: ${2 + 2}"

// Multiline strings (triple quotes)
val text = """
    Line 1
    Line 2
    Line 3
""".trimIndent()

// Raw strings (no escaping needed)
val regex = """C:\Users\name\Documents"""
val json = """
    {
        "name": "Alice",
        "age": 30
    }
""".trimIndent()

// String operations
val str = "Hello, World!"
val length = str.length
val upper = str.uppercase()  // HELLO, WORLD!
val lower = str.lowercase()  // hello, world!
val trimmed = "  hello  ".trim()  // "hello"
val substring = str.substring(0, 5)  // "Hello"
val replaced = str.replace("World", "Kotlin")  // "Hello, Kotlin!"
val contains = str.contains("World")  // true
val startsWith = str.startsWith("Hello")  // true
val endsWith = str.endsWith("!")  // true

// Split string
val parts = "a,b,c".split(",")  // List<String>

// String to number
val num = "42".toInt()
val double = "3.14".toDouble()
val numOrNull = "abc".toIntOrNull()  // null (safe conversion)

// String comparison
val str1 = "hello"
val str2 = "HELLO"
val equals = str1 == str2  // false
val equalsIgnoreCase = str1.equals(str2, ignoreCase = true)  // true

// StringBuilder
val sb = StringBuilder()
sb.append("Hello")
sb.append(" ")
sb.append("World")
val result = sb.toString()

// Join strings
val words = listOf("Kotlin", "is", "awesome")
val sentence = words.joinToString(" ")  // "Kotlin is awesome"
val csv = words.joinToString(", ")  // "Kotlin, is, awesome"
```

---

## Collections

### Lists

```kotlin
// Immutable list (read-only)
val numbers = listOf(1, 2, 3, 4, 5)
val names = listOf("Alice", "Bob", "Charlie")
val empty = emptyList<String>()

// Accessing elements
val first = numbers[0]  // 1
val second = numbers.get(1)  // 2
val firstOrNull = numbers.firstOrNull()  // 1
val lastOrNull = numbers.lastOrNull()  // 5
val elementOrNull = numbers.getOrNull(10)  // null

// List properties
val size = numbers.size
val isEmpty = numbers.isEmpty()
val isNotEmpty = numbers.isNotEmpty()

// Mutable list
val mutableList = mutableListOf(1, 2, 3)
mutableList.add(4)  // [1, 2, 3, 4]
mutableList.addAll(listOf(5, 6))  // [1, 2, 3, 4, 5, 6]
mutableList.remove(3)  // [1, 2, 4, 5, 6]
mutableList.removeAt(0)  // [2, 4, 5, 6]
mutableList.clear()  // []
mutableList[0] = 10  // Update element

// Create mutable list from immutable
val mutable = numbers.toMutableList()

// List operations
val contains = numbers.contains(3)  // true
val indexOf = numbers.indexOf(3)  // 2
val lastIndexOf = numbers.lastIndexOf(3)
val subList = numbers.subList(1, 4)  // [2, 3, 4]

// Iteration
for (num in numbers) {
    println(num)
}

for ((index, value) in numbers.withIndex()) {
    println("$index: $value")
}

numbers.forEach { println(it) }
numbers.forEachIndexed { index, value ->
    println("$index: $value")
}

// List of specific type
val strings: List<String> = listOf("a", "b", "c")
val ints: List<Int> = listOf(1, 2, 3)
```

### Sets

```kotlin
// Immutable set (read-only, unique elements)
val numbers = setOf(1, 2, 3, 4, 5)
val duplicates = setOf(1, 1, 2, 2, 3)  // [1, 2, 3]

// Mutable set
val mutableSet = mutableSetOf(1, 2, 3)
mutableSet.add(4)  // true (added)
mutableSet.add(2)  // false (already exists)
mutableSet.remove(3)  // true (removed)
mutableSet.clear()

// Set operations
val set1 = setOf(1, 2, 3, 4)
val set2 = setOf(3, 4, 5, 6)

val union = set1 union set2  // [1, 2, 3, 4, 5, 6]
val intersect = set1 intersect set2  // [3, 4]
val subtract = set1 subtract set2  // [1, 2]

// Check membership
val contains = 3 in numbers  // true
val notContains = 10 !in numbers  // true

// Convert list to set (removes duplicates)
val list = listOf(1, 2, 2, 3, 3, 3)
val uniqueSet = list.toSet()  // [1, 2, 3]
```

### Maps

```kotlin
// Immutable map (read-only)
val map = mapOf(
    "name" to "Alice",
    "age" to "30",
    "city" to "NYC"
)

val ages = mapOf(
    1 to "one",
    2 to "two",
    3 to "three"
)

// Accessing values
val name = map["name"]  // "Alice" (String?)
val age = map.get("age")  // "30"
val country = map["country"]  // null
val countryOrDefault = map.getOrDefault("country", "USA")

// Mutable map
val mutableMap = mutableMapOf(
    "name" to "Alice",
    "age" to "30"
)

mutableMap["email"] = "alice@example.com"  // Add/update
mutableMap.put("phone", "123-456-7890")
mutableMap.remove("age")
mutableMap.clear()

// Map operations
val size = map.size
val isEmpty = map.isEmpty()
val containsKey = map.containsKey("name")
val containsValue = map.containsValue("Alice")
val keys = map.keys  // Set<String>
val values = map.values  // Collection<String>
val entries = map.entries  // Set<Map.Entry<String, String>>

// Iteration
for ((key, value) in map) {
    println("$key: $value")
}

map.forEach { (key, value) ->
    println("$key: $value")
}

map.forEach { entry ->
    println("${entry.key}: ${entry.value}")
}

// Get or put
val computedValue = mutableMap.getOrPut("computed") {
    "default value"  // Only computed if key doesn't exist
}

// Filter map
val filtered = map.filter { (key, value) ->
    key.startsWith("n")
}

// Map values
val upperValues = map.mapValues { (_, value) ->
    value.uppercase()
}
```

---

## Control Flow

### If Expressions

In Kotlin, `if` is an expression that returns a value.

```kotlin
// Basic if
val age = 18
if (age >= 18) {
    println("Adult")
} else {
    println("Minor")
}

// If as expression
val status = if (age >= 18) "Adult" else "Minor"

// Multi-line if expression
val result = if (age < 13) {
    "Child"
} else if (age < 20) {
    "Teenager"
} else {
    "Adult"
}

// If with multiple conditions
if (age >= 18 && age < 65) {
    println("Working age")
}

// Null checks
val name: String? = "Alice"
if (name != null && name.length > 0) {
    println("Name is $name")
}

// Ranges
if (age in 13..19) {
    println("Teenager")
}

if (age !in 0..17) {
    println("Adult")
}
```

### When Expressions

`when` is Kotlin's replacement for switch, but more powerful.

```kotlin
// Basic when
val day = 3
when (day) {
    1 -> println("Monday")
    2 -> println("Tuesday")
    3 -> println("Wednesday")
    4 -> println("Thursday")
    5 -> println("Friday")
    6, 7 -> println("Weekend")
    else -> println("Invalid day")
}

// When as expression
val dayName = when (day) {
    1 -> "Monday"
    2 -> "Tuesday"
    3 -> "Wednesday"
    4 -> "Thursday"
    5 -> "Friday"
    6, 7 -> "Weekend"
    else -> "Invalid"
}

// When with ranges
val score = 85
val grade = when (score) {
    in 90..100 -> "A"
    in 80..89 -> "B"
    in 70..79 -> "C"
    in 60..69 -> "D"
    else -> "F"
}

// When with conditions
val x = 15
when {
    x < 0 -> println("Negative")
    x == 0 -> println("Zero")
    x > 0 && x < 10 -> println("Small positive")
    x >= 10 -> println("Large positive")
}

// When with type checking
fun describe(obj: Any): String = when (obj) {
    is String -> "String of length ${obj.length}"
    is Int -> "Integer: $obj"
    is Boolean -> "Boolean: $obj"
    is List<*> -> "List of size ${obj.size}"
    else -> "Unknown type"
}

// When with smart casts
fun process(value: Any) {
    when (value) {
        is String -> println(value.uppercase())  // Smart cast to String
        is Int -> println(value * 2)  // Smart cast to Int
        is List<*> -> println(value.size)  // Smart cast to List
    }
}

// When without argument
val temperature = 25
when {
    temperature < 0 -> println("Freezing")
    temperature < 15 -> println("Cold")
    temperature < 25 -> println("Moderate")
    else -> println("Hot")
}
```

### Loops

```kotlin
// For loop with range
for (i in 1..5) {
    println(i)  // 1, 2, 3, 4, 5
}

// For loop with until (exclusive)
for (i in 1 until 5) {
    println(i)  // 1, 2, 3, 4
}

// For loop with step
for (i in 1..10 step 2) {
    println(i)  // 1, 3, 5, 7, 9
}

// Downward range
for (i in 5 downTo 1) {
    println(i)  // 5, 4, 3, 2, 1
}

// Iterate over list
val names = listOf("Alice", "Bob", "Charlie")
for (name in names) {
    println(name)
}

// Iterate with index
for ((index, name) in names.withIndex()) {
    println("$index: $name")
}

// Iterate over map
val map = mapOf("a" to 1, "b" to 2, "c" to 3)
for ((key, value) in map) {
    println("$key: $value")
}

// While loop
var count = 0
while (count < 5) {
    println(count)
    count++
}

// Do-while loop
var x = 0
do {
    println(x)
    x++
} while (x < 5)

// Break and continue
for (i in 1..10) {
    if (i == 3) continue  // Skip 3
    if (i == 8) break  // Stop at 8
    println(i)
}

// Labeled break and continue
outer@ for (i in 1..5) {
    for (j in 1..5) {
        if (j == 3) break@outer  // Break outer loop
        println("$i, $j")
    }
}

// Repeat
repeat(3) {
    println("Hello")  // Prints 3 times
}

repeat(5) { index ->
    println("Iteration $index")
}

// Ranges
val range1 = 1..10  // 1 to 10 (inclusive)
val range2 = 1 until 10  // 1 to 9
val range3 = 10 downTo 1  // 10 to 1
val range4 = 1..10 step 2  // 1, 3, 5, 7, 9

val inRange = 5 in 1..10  // true
val notInRange = 15 !in 1..10  // true

// Character ranges
for (c in 'a'..'z') {
    print(c)  // abcdefghijklmnopqrstuvwxyz
}
```

---

## Functions

### Basic Functions

```kotlin
// Simple function
fun greet(name: String): String {
    return "Hello, $name!"
}

// Single-expression function
fun add(a: Int, b: Int): Int = a + b

// Function with inferred return type
fun multiply(a: Int, b: Int) = a * b

// Unit return type (like void in Java)
fun printMessage(message: String): Unit {
    println(message)
}

// Unit can be omitted
fun printMessage2(message: String) {
    println(message)
}

// Default parameters
fun greet(name: String = "World", greeting: String = "Hello"): String {
    return "$greeting, $name!"
}

val msg1 = greet()  // "Hello, World!"
val msg2 = greet("Alice")  // "Hello, Alice!"
val msg3 = greet("Bob", "Hi")  // "Hi, Bob!"

// Named arguments
val msg4 = greet(greeting = "Hey", name = "Charlie")

// Varargs (variable number of arguments)
fun sum(vararg numbers: Int): Int {
    return numbers.sum()
}

val result = sum(1, 2, 3, 4, 5)  // 15

// Spread operator
val nums = intArrayOf(1, 2, 3)
val total = sum(*nums)  // Spread array as varargs

// Multiple return values (using Pair/Triple)
fun getNameAndAge(): Pair<String, Int> {
    return Pair("Alice", 30)
}

val (name, age) = getNameAndAge()
println("$name is $age years old")

// Using Triple
fun getCoordinates(): Triple<Int, Int, Int> {
    return Triple(10, 20, 30)
}

val (x, y, z) = getCoordinates()

// Using data class for multiple returns
data class User(val name: String, val age: Int, val email: String)

fun getUser(): User {
    return User("Alice", 30, "alice@example.com")
}

// Nothing type (function never returns)
fun fail(message: String): Nothing {
    throw IllegalStateException(message)
}
```

### Higher-Order Functions and Lambdas

```kotlin
// Function type
val sum: (Int, Int) -> Int = { a, b -> a + b }
val result = sum(5, 3)  // 8

// Lambda expressions
val multiply = { a: Int, b: Int -> a * b }
val square = { x: Int -> x * x }
val greet = { name: String -> "Hello, $name!" }

// Lambda with single parameter (implicit 'it')
val double: (Int) -> Int = { it * 2 }
val isEven: (Int) -> Boolean = { it % 2 == 0 }

// Multi-line lambda
val calculate = { a: Int, b: Int ->
    val sum = a + b
    val product = a * b
    sum * product  // Last expression is returned
}

// Higher-order function (takes function as parameter)
fun operate(a: Int, b: Int, operation: (Int, Int) -> Int): Int {
    return operation(a, b)
}

val sum = operate(5, 3) { a, b -> a + b }  // 8
val product = operate(5, 3) { a, b -> a * b }  // 15

// Function as return type
fun getOperation(type: String): (Int, Int) -> Int {
    return when (type) {
        "add" -> { a, b -> a + b }
        "multiply" -> { a, b -> a * b }
        else -> { a, b -> 0 }
    }
}

val addFunc = getOperation("add")
val result = addFunc(5, 3)  // 8

// Anonymous function
val sum2 = fun(a: Int, b: Int): Int {
    return a + b
}

// Function references
fun isOdd(x: Int): Boolean = x % 2 == 1

val numbers = listOf(1, 2, 3, 4, 5)
val odds = numbers.filter(::isOdd)  // [1, 3, 5]

// Member function reference
val lengths = listOf("a", "abc", "abcdef").map(String::length)  // [1, 3, 6]

// Closure (accessing outer scope)
fun makeCounter(): () -> Int {
    var count = 0
    return {
        count++
        count
    }
}

val counter = makeCounter()
println(counter())  // 1
println(counter())  // 2
println(counter())  // 3
```

### Inline Functions

```kotlin
// Inline function (eliminates function call overhead)
inline fun <T> measureTime(block: () -> T): T {
    val start = System.currentTimeMillis()
    val result = block()
    val end = System.currentTimeMillis()
    println("Time taken: ${end - start}ms")
    return result
}

val result = measureTime {
    // Some expensive operation
    Thread.sleep(1000)
    42
}

// noinline (prevent specific lambda from being inlined)
inline fun foo(inlined: () -> Unit, noinline notInlined: () -> Unit) {
    // ...
}

// crossinline (lambda cannot use non-local returns)
inline fun bar(crossinline body: () -> Unit) {
    // ...
}
```

### Infix Functions

```kotlin
// Infix notation (call without dot and parentheses)
infix fun Int.times(str: String) = str.repeat(this)

val result = 3 times "Hello "  // "Hello Hello Hello "

// Another example
infix fun String.shouldBe(expected: String) {
    if (this != expected) {
        throw AssertionError("Expected $expected but got $this")
    }
}

"hello" shouldBe "hello"  // OK

// Common infix functions
val pair = "name" to "Alice"  // 'to' is infix
val range = 1 until 10  // 'until' is infix
```

### Operator Overloading

```kotlin
// Overload operators
data class Point(val x: Int, val y: Int) {
    operator fun plus(other: Point) = Point(x + other.x, y + other.y)
    operator fun minus(other: Point) = Point(x - other.x, y - other.y)
    operator fun times(scale: Int) = Point(x * scale, y * scale)
    operator fun unaryMinus() = Point(-x, -y)
    operator fun inc() = Point(x + 1, y + 1)
    operator fun get(index: Int) = when (index) {
        0 -> x
        1 -> y
        else -> throw IndexOutOfBoundsException()
    }
}

val p1 = Point(10, 20)
val p2 = Point(5, 10)
val p3 = p1 + p2  // Point(15, 30)
val p4 = p1 - p2  // Point(5, 10)
val p5 = p1 * 2  // Point(20, 40)
val p6 = -p1  // Point(-10, -20)
val x = p1[0]  // 10

// Other operators: compareTo, contains, iterator, rangeTo, etc.
```

---

## Object-Oriented Programming

### Classes and Objects

```kotlin
// Basic class
class Person {
    var name: String = ""
    var age: Int = 0

    fun greet() {
        println("Hello, I'm $name")
    }
}

val person = Person()
person.name = "Alice"
person.age = 30
person.greet()

// Primary constructor
class Person2(val name: String, var age: Int) {
    init {
        println("Person created: $name, $age")
    }

    fun greet() = "Hello, I'm $name"
}

val person2 = Person2("Alice", 30)

// Constructor with default values
class Person3(
    val name: String = "Unknown",
    var age: Int = 0,
    val email: String = "unknown@example.com"
)

val p1 = Person3("Alice", 30, "alice@example.com")
val p2 = Person3("Bob", 25)
val p3 = Person3(name = "Charlie")

// Secondary constructor
class Person4(val name: String) {
    var age: Int = 0
    var email: String = ""

    constructor(name: String, age: Int) : this(name) {
        this.age = age
    }

    constructor(name: String, age: Int, email: String) : this(name, age) {
        this.email = email
    }
}

// Properties with custom getters and setters
class Rectangle(val width: Int, val height: Int) {
    val area: Int
        get() = width * height

    var maxDimension: Int = 0
        get() = if (width > height) width else height
        set(value) {
            field = if (value >= 0) value else 0
        }
}

// Visibility modifiers
class Example {
    public val publicVar = 1  // Visible everywhere (default)
    private val privateVar = 2  // Visible in this class only
    protected val protectedVar = 3  // Visible in this class and subclasses
    internal val internalVar = 4  // Visible in same module
}
```

### Inheritance

```kotlin
// Open class (can be inherited)
open class Animal(val name: String) {
    open fun sound() {
        println("$name makes a sound")
    }

    open val category: String = "Animal"
}

// Inherit from Animal
class Dog(name: String, val breed: String) : Animal(name) {
    override fun sound() {
        println("$name barks")
    }

    override val category: String = "Mammal"

    fun fetch() {
        println("$name is fetching")
    }
}

class Cat(name: String) : Animal(name) {
    override fun sound() {
        println("$name meows")
    }
}

val dog = Dog("Buddy", "Golden Retriever")
dog.sound()  // "Buddy barks"
dog.fetch()

// Abstract classes
abstract class Shape {
    abstract val area: Double
    abstract fun perimeter(): Double

    // Concrete method
    fun describe() {
        println("Area: $area, Perimeter: ${perimeter()}")
    }
}

class Circle(val radius: Double) : Shape() {
    override val area: Double
        get() = Math.PI * radius * radius

    override fun perimeter(): Double = 2 * Math.PI * radius
}

class Rectangle2(val width: Double, val height: Double) : Shape() {
    override val area: Double
        get() = width * height

    override fun perimeter(): Double = 2 * (width + height)
}

// Prevent further inheritance
final class FinalClass {
    // Cannot be inherited
}
```

### Interfaces

```kotlin
// Interface definition
interface Drawable {
    fun draw()

    // Property in interface
    val color: String

    // Default implementation
    fun describe() {
        println("Drawing with color $color")
    }
}

interface Clickable {
    fun click()
    fun showOff() {
        println("I'm clickable!")
    }
}

// Implement interfaces
class Button : Drawable, Clickable {
    override val color: String = "Blue"

    override fun draw() {
        println("Drawing button")
    }

    override fun click() {
        println("Button clicked")
    }

    // Resolve conflicts when multiple interfaces have same method
    override fun showOff() {
        super<Clickable>.showOff()
    }
}

// Functional interface (SAM - Single Abstract Method)
fun interface StringProcessor {
    fun process(input: String): String
}

// Can be instantiated with lambda
val uppercase = StringProcessor { it.uppercase() }
val result = uppercase.process("hello")  // "HELLO"
```

### Data Classes

Data classes automatically generate `equals()`, `hashCode()`, `toString()`, `copy()`, and `componentN()` functions.

```kotlin
// Data class
data class User(
    val name: String,
    val age: Int,
    val email: String
)

val user1 = User("Alice", 30, "alice@example.com")
val user2 = User("Alice", 30, "alice@example.com")

// Automatically generated methods
println(user1.toString())  // User(name=Alice, age=30, email=alice@example.com)
println(user1 == user2)  // true (structural equality)
println(user1 === user2)  // false (referential equality)

// Copy with modifications
val user3 = user1.copy(age = 31)
val user4 = user1.copy(email = "newemail@example.com")

// Destructuring
val (name, age, email) = user1
println("$name is $age years old")

// Data classes can have body
data class Person(val name: String, val age: Int) {
    var nickname: String = ""

    fun greet() = "Hello, I'm $name"
}
```

### Sealed Classes

Sealed classes represent restricted class hierarchies.

```kotlin
// Sealed class (all subclasses must be in same file)
sealed class Result<out T> {
    data class Success<T>(val data: T) : Result<T>()
    data class Error(val message: String, val code: Int) : Result<Nothing>()
    object Loading : Result<Nothing>()
}

// Pattern matching with when
fun <T> handleResult(result: Result<T>) {
    when (result) {
        is Result.Success -> println("Success: ${result.data}")
        is Result.Error -> println("Error ${result.code}: ${result.message}")
        is Result.Loading -> println("Loading...")
        // No else needed - compiler knows all cases
    }
}

val success = Result.Success("Data loaded")
val error = Result.Error("Not found", 404)
val loading = Result.Loading

handleResult(success)
handleResult(error)

// Another example
sealed interface UiState {
    object Idle : UiState
    object Loading : UiState
    data class Success(val data: String) : UiState
    data class Error(val exception: Throwable) : UiState
}
```

### Object Declarations and Companion Objects

```kotlin
// Object (singleton)
object DatabaseConfig {
    val url = "jdbc:mysql://localhost:3306/mydb"
    val username = "admin"

    fun connect() {
        println("Connecting to $url")
    }
}

// Usage
DatabaseConfig.connect()
println(DatabaseConfig.url)

// Object expression (anonymous object)
val clickListener = object : Clickable {
    override fun click() {
        println("Clicked!")
    }
}

// Companion object (similar to static members in Java)
class MyClass {
    companion object {
        const val CONSTANT = 42

        fun create(): MyClass {
            return MyClass()
        }
    }

    fun instanceMethod() {
        println("Instance method")
    }
}

// Usage
val constant = MyClass.CONSTANT
val instance = MyClass.create()
instance.instanceMethod()

// Named companion object
class User {
    companion object Factory {
        fun create(name: String): User {
            return User()
        }
    }
}

val user = User.create("Alice")
// or
val user2 = User.Factory.create("Bob")

// Companion object implementing interface
interface JsonSerializer<T> {
    fun toJson(obj: T): String
}

class Person(val name: String, val age: Int) {
    companion object : JsonSerializer<Person> {
        override fun toJson(obj: Person): String {
            return """{"name":"${obj.name}","age":${obj.age}}"""
        }
    }
}

val json = Person.toJson(Person("Alice", 30))
```

### Nested and Inner Classes

```kotlin
// Nested class (static by default)
class Outer {
    private val bar: Int = 1

    class Nested {
        fun foo() = 2
        // Cannot access bar (no reference to Outer)
    }
}

val demo = Outer.Nested().foo()

// Inner class (has reference to outer)
class Outer2 {
    private val bar: Int = 1

    inner class Inner {
        fun foo() = bar  // Can access bar
        fun getOuter() = this@Outer2
    }
}

val outer = Outer2()
val inner = outer.Inner()
val result = inner.foo()  // 1
```

### Enum Classes

```kotlin
// Basic enum
enum class Direction {
    NORTH, SOUTH, EAST, WEST
}

val direction = Direction.NORTH

// Enum with properties
enum class Color(val rgb: Int) {
    RED(0xFF0000),
    GREEN(0x00FF00),
    BLUE(0x0000FF)
}

val red = Color.RED
val rgb = red.rgb

// Enum with methods
enum class Operation {
    ADD {
        override fun apply(x: Int, y: Int) = x + y
    },
    SUBTRACT {
        override fun apply(x: Int, y: Int) = x - y
    },
    MULTIPLY {
        override fun apply(x: Int, y: Int) = x * y
    },
    DIVIDE {
        override fun apply(x: Int, y: Int) = x / y
    };

    abstract fun apply(x: Int, y: Int): Int
}

val result = Operation.ADD.apply(5, 3)  // 8

// Enum properties and methods
enum class Status(val description: String) {
    PENDING("Waiting for approval"),
    APPROVED("Request approved"),
    REJECTED("Request rejected");

    fun isComplete() = this != PENDING
}

// Iterate over enum values
for (status in Status.values()) {
    println("${status.name}: ${status.description}")
}

// Get enum by name
val status = Status.valueOf("APPROVED")

// Modern way (Kotlin 1.9+)
val statuses = Status.entries
```

---

## Extension Functions

Extension functions allow you to add new functions to existing classes without modifying their source code.

```kotlin
// Extension function
fun String.removeWhitespace(): String {
    return this.replace(" ", "")
}

val text = "Hello World"
val result = text.removeWhitespace()  // "HelloWorld"

// Extension with receiver
fun Int.isEven(): Boolean = this % 2 == 0
fun Int.isOdd(): Boolean = this % 2 == 1

println(4.isEven())  // true
println(5.isOdd())  // true

// Extension properties
val String.firstChar: Char
    get() = if (this.isNotEmpty()) this[0] else ' '

val String.lastChar: Char
    get() = if (this.isNotEmpty()) this[this.length - 1] else ' '

println("Hello".firstChar)  // 'H'
println("World".lastChar)  // 'd'

// Extensions on nullable types
fun String?.isNullOrBlank(): Boolean {
    return this == null || this.isBlank()
}

val nullString: String? = null
println(nullString.isNullOrBlank())  // true

// Extension function for collections
fun <T> List<T>.secondOrNull(): T? {
    return if (this.size >= 2) this[1] else null
}

val list = listOf(1, 2, 3)
println(list.secondOrNull())  // 2

// Generic extension function
fun <T> T.applyIf(condition: Boolean, block: T.() -> Unit): T {
    if (condition) {
        this.block()
    }
    return this
}

val builder = StringBuilder()
    .append("Hello")
    .applyIf(true) { append(" World") }
    .applyIf(false) { append(" Kotlin") }
    .toString()  // "Hello World"

// Extension functions on companion objects
class MyClass {
    companion object { }
}

fun MyClass.Companion.create(): MyClass {
    return MyClass()
}

val instance = MyClass.create()
```

---

## Scope Functions

Kotlin provides scope functions to execute code blocks within the context of an object.

```kotlin
// let - Execute lambda on object, returns lambda result
val name: String? = "Alice"
val length = name?.let {
    println("Name is $it")
    it.length  // Returns this
} ?: 0

val numbers = listOf(1, 2, 3)
numbers.let {
    println("List size: ${it.size}")
    it.filter { num -> num > 1 }
}

// run - Execute lambda on object, returns lambda result
val person = Person("Alice", 30)
val greeting = person.run {
    println("Name: $name")
    println("Age: $age")
    "Hello, $name"  // Returns this
}

// with - Non-extension function, returns lambda result
val result = with(person) {
    println("Name: $name")
    println("Age: $age")
    "Processed"
}

// apply - Configure object, returns the object itself
val person2 = Person("Bob", 25).apply {
    age = 26
    email = "bob@example.com"
}

val list = mutableListOf<String>().apply {
    add("One")
    add("Two")
    add("Three")
}

// also - Perform side effects, returns the object itself
val numbers2 = mutableListOf(1, 2, 3).also {
    println("Initial list: $it")
}.also {
    it.add(4)
}.also {
    println("After adding: $it")
}

// takeIf - Returns object if predicate is true, else null
val positiveNumber = 42.takeIf { it > 0 }  // 42
val negativeNumber = (-5).takeIf { it > 0 }  // null

val validName = "Alice".takeIf { it.length > 3 }  // "Alice"

// takeUnless - Returns object if predicate is false, else null
val shortName = "Al".takeUnless { it.length > 3 }  // "Al"
val longName = "Alice".takeUnless { it.length > 3 }  // null

// Chaining scope functions
val result = listOf(1, 2, 3, 4, 5)
    .filter { it > 2 }
    .also { println("Filtered: $it") }
    .map { it * 2 }
    .also { println("Mapped: $it") }
    .sum()
    .also { println("Sum: $it") }

// Practical example
data class User(var name: String, var age: Int, var email: String = "")

val user = User("Alice", 30).apply {
    email = "alice@example.com"
}.also {
    println("Created user: $it")
}.takeIf {
    it.age >= 18
}?.let {
    "Valid adult user: ${it.name}"
} ?: "Invalid user"
```

---

## Delegation

### Class Delegation

```kotlin
interface Base {
    fun print()
    fun printMessage(message: String)
}

class BaseImpl(val x: Int) : Base {
    override fun print() {
        println(x)
    }

    override fun printMessage(message: String) {
        println(message)
    }
}

// Delegate to BaseImpl
class Derived(b: Base) : Base by b {
    // Can override if needed
    override fun printMessage(message: String) {
        println("Derived: $message")
    }
}

val base = BaseImpl(10)
val derived = Derived(base)
derived.print()  // 10 (delegated to base)
derived.printMessage("Hello")  // "Derived: Hello" (overridden)
```

### Property Delegation

```kotlin
import kotlin.properties.Delegates

class User {
    // Lazy property (initialized on first access)
    val heavyData: String by lazy {
        println("Computing heavy data...")
        "Heavy Data"
    }

    // Observable property (notified on change)
    var name: String by Delegates.observable("Initial") { prop, old, new ->
        println("${prop.name} changed from $old to $new")
    }

    // Vetoable property (can reject changes)
    var age: Int by Delegates.vetoable(0) { prop, old, new ->
        new >= 0  // Only allow non-negative ages
    }

    // Not-null property (must be initialized before use)
    var email: String by Delegates.notNull()

    // Delegate to another property
    var nickname: String = ""
    var displayName: String by this::nickname
}

val user = User()
user.name = "Alice"  // Prints: name changed from Initial to Alice
user.age = 30  // OK
user.age = -5  // Rejected (age remains 30)
user.email = "alice@example.com"

// First access triggers lazy initialization
println(user.heavyData)  // Prints: "Computing heavy data..." then "Heavy Data"
println(user.heavyData)  // Just prints: "Heavy Data"

// Custom delegation
import kotlin.reflect.KProperty

class Delegate {
    operator fun getValue(thisRef: Any?, property: KProperty<*>): String {
        return "$thisRef, thank you for delegating '${property.name}' to me!"
    }

    operator fun setValue(thisRef: Any?, property: KProperty<*>, value: String) {
        println("$value has been assigned to '${property.name}' in $thisRef.")
    }
}

class Example {
    var p: String by Delegate()
}

// Map delegation
class UserFromMap(map: Map<String, Any?>) {
    val name: String by map
    val age: Int by map
}

val user2 = UserFromMap(mapOf(
    "name" to "Bob",
    "age" to 25
))
println(user2.name)  // Bob
println(user2.age)  // 25
```

---

## Collection Operations

Kotlin provides extensive collection operations (similar to Java Streams).

```kotlin
val numbers = listOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

// Filter
val evens = numbers.filter { it % 2 == 0 }  // [2, 4, 6, 8, 10]
val odds = numbers.filterNot { it % 2 == 0 }  // [1, 3, 5, 7, 9]
val greaterThan5 = numbers.filter { it > 5 }  // [6, 7, 8, 9, 10]

// Map (transform)
val squared = numbers.map { it * it }  // [1, 4, 9, 16, 25, ...]
val strings = numbers.map { "Number: $it" }

// FlatMap (flatten nested collections)
val nested = listOf(listOf(1, 2), listOf(3, 4), listOf(5, 6))
val flat = nested.flatten()  // [1, 2, 3, 4, 5, 6]

val words = listOf("hello", "world")
val chars = words.flatMap { it.toList() }  // [h, e, l, l, o, w, o, r, l, d]

// Distinct
val duplicates = listOf(1, 2, 2, 3, 3, 3, 4)
val unique = duplicates.distinct()  // [1, 2, 3, 4]

// DistinctBy
data class Person(val name: String, val age: Int)
val people = listOf(
    Person("Alice", 30),
    Person("Bob", 25),
    Person("Alice", 35)
)
val uniqueNames = people.distinctBy { it.name }  // [Alice(30), Bob(25)]

// Take and drop
val first3 = numbers.take(3)  // [1, 2, 3]
val last3 = numbers.takeLast(3)  // [8, 9, 10]
val without3 = numbers.drop(3)  // [4, 5, 6, 7, 8, 9, 10]
val takeWhileLessThan5 = numbers.takeWhile { it < 5 }  // [1, 2, 3, 4]

// Sorted
val sorted = listOf(5, 2, 8, 1, 9).sorted()  // [1, 2, 5, 8, 9]
val sortedDesc = numbers.sortedDescending()
val sortedByAge = people.sortedBy { it.age }
val sortedByName = people.sortedByDescending { it.name }

// GroupBy
val grouped = numbers.groupBy { it % 3 }
// {1=[1, 4, 7, 10], 2=[2, 5, 8], 0=[3, 6, 9]}

val groupedByAge = people.groupBy { it.age }

// Partition (split by predicate)
val (evens2, odds2) = numbers.partition { it % 2 == 0 }

// Reduce and fold
val sum = numbers.reduce { acc, n -> acc + n }  // 55
val product = numbers.fold(1) { acc, n -> acc * n }

val concatenated = listOf("a", "b", "c").reduce { acc, s -> acc + s }  // "abc"

// Any, all, none
val hasEven = numbers.any { it % 2 == 0 }  // true
val allPositive = numbers.all { it > 0 }  // true
val noneNegative = numbers.none { it < 0 }  // true

// Find
val firstEven = numbers.find { it % 2 == 0 }  // 2 (or null if not found)
val firstOrNull = numbers.firstOrNull { it > 100 }  // null
val lastOdd = numbers.lastOrNull { it % 2 == 1 }  // 9

// Count
val evenCount = numbers.count { it % 2 == 0 }  // 5

// Sum, average, min, max
val total = numbers.sum()  // 55
val average = numbers.average()  // 5.5
val min = numbers.minOrNull()  // 1
val max = numbers.maxOrNull()  // 10

// SumOf, minOf, maxOf
val totalAge = people.sumOf { it.age }
val minAge = people.minOf { it.age }
val maxAge = people.maxOf { it.age }

// Associate (create map from list)
val nameToAge = people.associate { it.name to it.age }
val ageToName = people.associateBy { it.age }
val nameToUpper = people.associateWith { it.name.uppercase() }

// Zip (combine two lists)
val names = listOf("Alice", "Bob", "Charlie")
val ages = listOf(30, 25, 35)
val pairs = names.zip(ages)  // [(Alice, 30), (Bob, 25), (Charlie, 35)]

val combined = names.zip(ages) { name, age ->
    "$name is $age years old"
}

// Chunked (split into sublists)
val chunks = numbers.chunked(3)  // [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

// Windowed (sliding window)
val windows = numbers.windowed(3)  // [[1,2,3], [2,3,4], [3,4,5], ...]

// JoinToString
val csv = numbers.joinToString(", ")  // "1, 2, 3, 4, 5, ..."
val custom = numbers.joinToString(
    separator = " | ",
    prefix = "[",
    postfix = "]",
    limit = 5,
    truncated = "..."
) { "No.$it" }  // "[No.1 | No.2 | No.3 | No.4 | No.5 | ...]"

// Sequences (lazy evaluation)
val sequence = numbers.asSequence()
    .filter { println("Filter: $it"); it % 2 == 0 }
    .map { println("Map: $it"); it * it }
    .toList()  // Operations only executed here
```

---

## Coroutines

Coroutines provide a way to write asynchronous code that looks synchronous.

```kotlin
import kotlinx.coroutines.*

// Basic coroutine launch
fun main() = runBlocking {
    launch {
        delay(1000L)
        println("World!")
    }
    println("Hello,")
}

// Async/await
suspend fun fetchData(): String {
    delay(1000L)
    return "Data"
}

fun main2() = runBlocking {
    val deferred = async {
        fetchData()
    }
    println("Fetching...")
    val result = deferred.await()
    println("Result: $result")
}

// Multiple async operations
fun main3() = runBlocking {
    val time = measureTimeMillis {
        val one = async { fetchOne() }
        val two = async { fetchTwo() }
        println("The answer is ${one.await() + two.await()}")
    }
    println("Completed in $time ms")
}

suspend fun fetchOne(): Int {
    delay(1000L)
    return 1
}

suspend fun fetchTwo(): Int {
    delay(1000L)
    return 2
}

// Coroutine scope
fun main4() = runBlocking {
    launch {
        repeat(5) { i ->
            println("Coroutine A: $i")
            delay(500L)
        }
    }

    launch {
        repeat(5) { i ->
            println("Coroutine B: $i")
            delay(300L)
        }
    }

    delay(3000L)
}

// Structured concurrency
suspend fun doWork() = coroutineScope {
    launch {
        delay(1000L)
        println("Task 1")
    }
    launch {
        delay(2000L)
        println("Task 2")
    }
    println("All tasks started")
}

// Flow (cold asynchronous stream)
fun getNumbers(): Flow<Int> = flow {
    for (i in 1..5) {
        delay(100)
        emit(i)
    }
}

fun main5() = runBlocking {
    getNumbers().collect { value ->
        println(value)
    }
}

// Flow operators
fun main6() = runBlocking {
    (1..10).asFlow()
        .filter { it % 2 == 0 }
        .map { it * it }
        .collect { println(it) }
}

// Exception handling
fun main7() = runBlocking {
    val job = launch {
        try {
            repeat(1000) { i ->
                println("Job: $i")
                delay(500L)
            }
        } catch (e: CancellationException) {
            println("Job cancelled")
        } finally {
            println("Job cleanup")
        }
    }

    delay(1300L)
    println("Cancelling job")
    job.cancelAndJoin()
    println("Main done")
}

// Timeout
fun main8() = runBlocking {
    withTimeout(1300L) {
        repeat(1000) { i ->
            println("Task: $i")
            delay(500L)
        }
    }
}

// withContext (switch context)
fun main9() = runBlocking {
    launch(Dispatchers.Default) {
        println("Default: ${Thread.currentThread().name}")

        withContext(Dispatchers.IO) {
            println("IO: ${Thread.currentThread().name}")
            delay(1000L)
        }

        withContext(Dispatchers.Main) {
            println("Main: ${Thread.currentThread().name}")
        }
    }
}
```

---

## Generics

```kotlin
// Generic class
class Box<T>(val value: T)

val intBox = Box(42)
val stringBox = Box("Hello")
val int = intBox.value  // Int
val str = stringBox.value  // String

// Generic function
fun <T> singletonList(item: T): List<T> {
    return listOf(item)
}

val list = singletonList(42)  // List<Int>
val list2 = singletonList("Hello")  // List<String>

// Multiple type parameters
class Pair<K, V>(val key: K, val value: V)

val pair = Pair("name", "Alice")

// Type constraints
fun <T : Comparable<T>> sort(list: List<T>): List<T> {
    return list.sorted()
}

val sorted = sort(listOf(3, 1, 4, 1, 5))

// Multiple constraints
fun <T> copyWhenGreater(list: List<T>, threshold: T): List<T>
    where T : Comparable<T>, T : Number {
    return list.filter { it > threshold }
}

// Variance: out (covariant)
interface Producer<out T> {
    fun produce(): T
}

class StringProducer : Producer<String> {
    override fun produce() = "Hello"
}

val producer: Producer<Any> = StringProducer()  // OK (covariant)

// Variance: in (contravariant)
interface Consumer<in T> {
    fun consume(item: T)
}

class AnyConsumer : Consumer<Any> {
    override fun consume(item: Any) {
        println(item)
    }
}

val consumer: Consumer<String> = AnyConsumer()  // OK (contravariant)

// Star projection
fun printList(list: List<*>) {
    for (item in list) {
        println(item)
    }
}

// Reified type parameters (with inline)
inline fun <reified T> isA(value: Any): Boolean {
    return value is T
}

println(isA<String>("Hello"))  // true
println(isA<Int>("Hello"))  // false

inline fun <reified T> parseJson(json: String): T {
    // Can access T::class at runtime
    return when (T::class) {
        String::class -> json as T
        Int::class -> json.toInt() as T
        else -> throw IllegalArgumentException()
    }
}
```

---

## Error Handling

```kotlin
// Try-catch
try {
    val result = 10 / 0
} catch (e: ArithmeticException) {
    println("Cannot divide by zero!")
} catch (e: Exception) {
    println("General error: ${e.message}")
} finally {
    println("Cleanup code")
}

// Try as expression
val result = try {
    "123".toInt()
} catch (e: NumberFormatException) {
    0
}

// Nothing type
fun fail(message: String): Nothing {
    throw IllegalArgumentException(message)
}

// Require (for arguments)
fun setAge(age: Int) {
    require(age >= 0) { "Age cannot be negative" }
    // ...
}

// Check (for state)
fun process() {
    check(isInitialized) { "Not initialized" }
    // ...
}

// requireNotNull
fun processUser(user: User?) {
    val nonNullUser = requireNotNull(user) { "User cannot be null" }
    // nonNullUser is now User (not nullable)
}

// checkNotNull
val value = checkNotNull(nullableValue) { "Value is null" }

// Result type (Kotlin 1.3+)
fun divide(a: Int, b: Int): Result<Int> {
    return if (b == 0) {
        Result.failure(ArithmeticException("Division by zero"))
    } else {
        Result.success(a / b)
    }
}

val result = divide(10, 2)
result.onSuccess { println("Result: $it") }
result.onFailure { println("Error: ${it.message}") }

val value = result.getOrNull()  // Int? (null on failure)
val valueOrDefault = result.getOrDefault(0)
val valueOrElse = result.getOrElse { 0 }

// runCatching (wraps exceptions in Result)
val result2 = runCatching {
    "123".toInt()
}

val result3 = runCatching {
    "abc".toInt()
}.onFailure {
    println("Failed: ${it.message}")
}.getOrDefault(0)

// Custom exceptions
class InvalidUserException(message: String) : Exception(message)
class UserNotFoundException(val userId: Int) : Exception("User $userId not found")

fun findUser(id: Int): User {
    if (id < 0) {
        throw InvalidUserException("Invalid user ID")
    }
    if (id > 1000) {
        throw UserNotFoundException(id)
    }
    return User("User$id", 30)
}

// Using exceptions
try {
    val user = findUser(2000)
} catch (e: UserNotFoundException) {
    println("User ${e.userId} not found")
} catch (e: InvalidUserException) {
    println("Invalid user: ${e.message}")
}
```

---

## File I/O

```kotlin
import java.io.File

// Read entire file
val content = File("file.txt").readText()
val lines = File("file.txt").readLines()  // List<String>
val bytes = File("file.txt").readBytes()

// Read with specific encoding
val utf8Content = File("file.txt").readText(Charsets.UTF_8)

// Read line by line (efficient for large files)
File("file.txt").forEachLine { line ->
    println(line)
}

File("file.txt").useLines { lines ->
    lines.forEach { println(it) }
}

// Write to file
File("output.txt").writeText("Hello, World!")
File("output.txt").writeBytes(byteArrayOf(1, 2, 3))

// Append to file
File("output.txt").appendText("\nNew line")

// Write lines
val lines = listOf("Line 1", "Line 2", "Line 3")
File("output.txt").writeText(lines.joinToString("\n"))

// BufferedReader/Writer
File("file.txt").bufferedReader().use { reader ->
    var line = reader.readLine()
    while (line != null) {
        println(line)
        line = reader.readLine()
    }
}

File("output.txt").bufferedWriter().use { writer ->
    writer.write("Line 1\n")
    writer.write("Line 2\n")
}

// PrintWriter
File("output.txt").printWriter().use { writer ->
    writer.println("Line 1")
    writer.println("Line 2")
}

// File operations
val file = File("path/to/file.txt")
val exists = file.exists()
val isFile = file.isFile
val isDirectory = file.isDirectory
val canRead = file.canRead()
val canWrite = file.canWrite()
val size = file.length()
val name = file.name
val path = file.path
val absolutePath = file.absolutePath
val parent = file.parent

// Create/delete
file.createNewFile()
file.delete()
file.mkdir()  // Create directory
file.mkdirs()  // Create directory and parents
file.deleteRecursively()  // Delete directory and contents

// List files
val dir = File("directory")
val files = dir.listFiles()  // Array<File>?
val fileNames = dir.list()  // Array<String>?

dir.walk().forEach { file ->
    println(file.path)
}

// Copy/move
val source = File("source.txt")
val dest = File("dest.txt")
source.copyTo(dest, overwrite = true)

// Working with paths
val file2 = File("dir", "file.txt")  // dir/file.txt
val file3 = File(File("dir"), "file.txt")

// Temp files
val tempFile = File.createTempFile("prefix", ".txt")
tempFile.deleteOnExit()
```

---

## Common Patterns

### Singleton Pattern

```kotlin
object Singleton {
    fun doSomething() {
        println("Singleton method")
    }
}

// Usage
Singleton.doSomething()
```

### Factory Pattern

```kotlin
interface Animal {
    fun sound(): String
}

class Dog : Animal {
    override fun sound() = "Woof!"
}

class Cat : Animal {
    override fun sound() = "Meow!"
}

object AnimalFactory {
    fun create(type: String): Animal {
        return when (type) {
            "dog" -> Dog()
            "cat" -> Cat()
            else -> throw IllegalArgumentException("Unknown animal type")
        }
    }
}

// Usage
val dog = AnimalFactory.create("dog")
println(dog.sound())
```

### Builder Pattern

```kotlin
class Pizza private constructor(
    val size: String,
    val cheese: Boolean,
    val pepperoni: Boolean,
    val mushrooms: Boolean
) {
    class Builder {
        private var size: String = "Medium"
        private var cheese: Boolean = false
        private var pepperoni: Boolean = false
        private var mushrooms: Boolean = false

        fun size(size: String) = apply { this.size = size }
        fun cheese(value: Boolean = true) = apply { this.cheese = value }
        fun pepperoni(value: Boolean = true) = apply { this.pepperoni = value }
        fun mushrooms(value: Boolean = true) = apply { this.mushrooms = value }

        fun build() = Pizza(size, cheese, pepperoni, mushrooms)
    }
}

// Usage
val pizza = Pizza.Builder()
    .size("Large")
    .cheese()
    .pepperoni()
    .build()

// Or use data class with defaults
data class Pizza2(
    val size: String = "Medium",
    val cheese: Boolean = false,
    val pepperoni: Boolean = false,
    val mushrooms: Boolean = false
)

val pizza2 = Pizza2(
    size = "Large",
    cheese = true,
    pepperoni = true
)
```

### Observer Pattern

```kotlin
interface Observer {
    fun update(data: String)
}

class Subject {
    private val observers = mutableListOf<Observer>()

    fun attach(observer: Observer) {
        observers.add(observer)
    }

    fun detach(observer: Observer) {
        observers.remove(observer)
    }

    fun notify(data: String) {
        observers.forEach { it.update(data) }
    }
}

class ConcreteObserver(val name: String) : Observer {
    override fun update(data: String) {
        println("$name received: $data")
    }
}

// Usage
val subject = Subject()
val observer1 = ConcreteObserver("Observer 1")
val observer2 = ConcreteObserver("Observer 2")

subject.attach(observer1)
subject.attach(observer2)
subject.notify("Hello!")
```

### Strategy Pattern

```kotlin
interface PaymentStrategy {
    fun pay(amount: Double)
}

class CreditCardPayment : PaymentStrategy {
    override fun pay(amount: Double) {
        println("Paid $$amount with credit card")
    }
}

class PayPalPayment : PaymentStrategy {
    override fun pay(amount: Double) {
        println("Paid $$amount with PayPal")
    }
}

class ShoppingCart {
    private var paymentStrategy: PaymentStrategy? = null

    fun setPaymentStrategy(strategy: PaymentStrategy) {
        paymentStrategy = strategy
    }

    fun checkout(amount: Double) {
        paymentStrategy?.pay(amount)
    }
}

// Usage
val cart = ShoppingCart()
cart.setPaymentStrategy(CreditCardPayment())
cart.checkout(100.0)

cart.setPaymentStrategy(PayPalPayment())
cart.checkout(50.0)
```

---

## Testing

```kotlin
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.AfterEach

class CalculatorTest {
    private lateinit var calculator: Calculator

    @BeforeEach
    fun setup() {
        calculator = Calculator()
    }

    @Test
    fun `test addition`() {
        val result = calculator.add(2, 3)
        assertEquals(5, result)
    }

    @Test
    fun `test subtraction`() {
        val result = calculator.subtract(5, 3)
        assertEquals(2, result)
    }

    @Test
    fun `test division by zero throws exception`() {
        assertThrows<ArithmeticException> {
            calculator.divide(10, 0)
        }
    }

    @Test
    fun `test list is not empty`() {
        val list = listOf(1, 2, 3)
        assertTrue(list.isNotEmpty())
        assertFalse(list.isEmpty())
    }

    @Test
    fun `test nullable value`() {
        val value: String? = null
        assertNull(value)

        val nonNull = "Hello"
        assertNotNull(nonNull)
    }

    @AfterEach
    fun teardown() {
        // Cleanup
    }
}

// Kotlin test assertions
import kotlin.test.*

class StringUtilsTest {
    @Test
    fun testUppercase() {
        assertEquals("HELLO", "hello".uppercase())
    }

    @Test
    fun testContains() {
        assertTrue("Hello World".contains("World"))
    }
}
```

---

## Best Practices

1. **Use val over var**
   ```kotlin
   val immutable = "Cannot change"  // Preferred
   var mutable = "Can change"  // Use when necessary
   ```

2. **Leverage null safety**
   ```kotlin
   val name: String? = getName()
   val length = name?.length ?: 0
   ```

3. **Use data classes**
   ```kotlin
   data class User(val name: String, val age: Int)
   ```

4. **Prefer extension functions**
   ```kotlin
   fun String.removeWhitespace() = replace(" ", "")
   ```

5. **Use scope functions appropriately**
   ```kotlin
   val user = User("Alice", 30).apply {
       email = "alice@example.com"
   }
   ```

6. **Use when instead of if-else chains**
   ```kotlin
   when (value) {
       1 -> println("One")
       2 -> println("Two")
       else -> println("Other")
   }
   ```

7. **Use named arguments for clarity**
   ```kotlin
   createUser(name = "Alice", age = 30, email = "alice@example.com")
   ```

8. **Use default parameters**
   ```kotlin
   fun greet(name: String = "World") = "Hello, $name!"
   ```

9. **Use collections operations**
   ```kotlin
   val evens = numbers.filter { it % 2 == 0 }
   ```

10. **Use sealed classes for restricted hierarchies**
    ```kotlin
    sealed class Result {
        data class Success(val data: String) : Result()
        data class Error(val error: String) : Result()
    }
    ```

---

## Common Libraries and Frameworks

- **Ktor**: Asynchronous web framework
- **Exposed**: SQL framework
- **kotlinx.serialization**: JSON serialization
- **Koin**: Dependency injection
- **Coroutines**: Async programming
- **Arrow**: Functional programming
- **MockK**: Mocking library for testing
- **Kotest**: Testing framework
- **Kotlin Multiplatform**: Share code across platforms
- **Compose**: UI framework (Android/Desktop/Web)

---

## Kotlin vs Java Quick Reference

| Feature | Java | Kotlin |
|---------|------|--------|
| Variable | `String name = "Alice";` | `val name = "Alice"` |
| Mutable | `String name = "Alice";` | `var name = "Alice"` |
| Nullable | `String name = null;` | `var name: String? = null` |
| String template | `"Hello " + name` | `"Hello $name"` |
| Ternary | `x > 0 ? 1 : 0` | `if (x > 0) 1 else 0` |
| Switch | `switch(x) {...}` | `when(x) {...}` |
| Static | `static void foo()` | `companion object { fun foo() }` |
| Singleton | `enum` or manual | `object Singleton` |
| Data class | Lombok or manual | `data class User(...)` |
| Lambda | `(x, y) -> x + y` | `{ x, y -> x + y }` |
| Extension | Not available | `fun String.ext() {...}` |
| Null check | `if (x != null) x.length()` | `x?.length` |
| Smart cast | Manual cast needed | Automatic after null check |
