# Go Programming

## Overview

Go (Golang) is a statically typed, compiled programming language designed at Google. It's known for its simplicity, efficiency, and excellent support for concurrent programming.

**Key Features:**
- Fast compilation and execution
- Built-in concurrency (goroutines and channels)
- Garbage collection
- Strong static typing with type inference
- Simple and clean syntax
- Excellent standard library
- Cross-platform compilation

---

## Basic Syntax

### Variables and Data Types

```go
package main

import "fmt"

func main() {
    // Variable declaration
    var name string = "Alice"
    var age int = 30

    // Short declaration (type inference)
    city := "NYC"
    isActive := true

    // Multiple declarations
    var x, y, z int = 1, 2, 3
    a, b := 10, 20

    // Constants
    const PI = 3.14159
    const MaxSize = 100

    // Zero values (default values)
    var num int        // 0
    var str string     // ""
    var flag bool      // false
    var ptr *int       // nil

    fmt.Println(name, age, city, isActive)
}
```

### Data Types

```go
// Basic types
var i int = 42           // Platform-dependent (32 or 64 bit)
var i8 int8 = 127        // 8-bit
var i16 int16 = 32767    // 16-bit
var i32 int32 = 2147483647  // 32-bit (rune alias)
var i64 int64 = 9223372036854775807  // 64-bit

var u uint = 42          // Unsigned, platform-dependent
var u8 uint8 = 255       // 8-bit (byte alias)

var f32 float32 = 3.14   // 32-bit float
var f64 float64 = 3.14159  // 64-bit float

var c64 complex64 = 1 + 2i
var c128 complex128 = 1 + 2i

var b bool = true
var r rune = 'A'         // Unicode code point (int32)
var by byte = 65         // Alias for uint8

var str string = "Hello, 世界"

// Type conversion
var x int = 42
var y float64 = float64(x)
var z uint = uint(x)
```

### Strings

```go
// String operations
s1 := "Hello"
s2 := "World"

// Concatenation
full := s1 + " " + s2

// Length (bytes, not runes)
length := len(s1)

// Accessing bytes
firstByte := s1[0]

// Substrings
sub := s1[1:4]  // "ell"

// String comparison
if s1 == s2 {
    fmt.Println("Equal")
}

// Multi-line strings
multiline := `This is a
multi-line
string`

// String iteration
for i, char := range "Hello" {
    fmt.Printf("%d: %c\n", i, char)
}

// String formatting
import "fmt"
formatted := fmt.Sprintf("Name: %s, Age: %d", "Alice", 30)

// String conversion
import "strconv"
numStr := strconv.Itoa(42)        // int to string
num, err := strconv.Atoi("42")    // string to int
```

---

## Arrays and Slices

### Arrays

```go
// Fixed-size arrays
var arr [5]int
arr[0] = 1

// Array literal
numbers := [5]int{1, 2, 3, 4, 5}

// Compiler counts length
auto := [...]int{1, 2, 3, 4}

// Multi-dimensional arrays
matrix := [3][3]int{
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
}

// Array length
length := len(numbers)

// Iterate over array
for i, v := range numbers {
    fmt.Printf("%d: %d\n", i, v)
}
```

### Slices (Dynamic Arrays)

```go
// Creating slices
var slice []int                    // nil slice
slice = []int{1, 2, 3, 4, 5}      // slice literal
slice = make([]int, 5)             // length 5, all zeros
slice = make([]int, 5, 10)         // length 5, capacity 10

// Append to slice
slice = append(slice, 6)
slice = append(slice, 7, 8, 9)

// Slice operations
arr := []int{1, 2, 3, 4, 5}
sub := arr[1:4]        // [2, 3, 4]
first := arr[:3]       // [1, 2, 3]
last := arr[3:]        // [4, 5]

// Length and capacity
len := len(slice)
cap := cap(slice)

// Copy slices
src := []int{1, 2, 3}
dst := make([]int, len(src))
copy(dst, src)

// 2D slices
matrix := [][]int{
    {1, 2, 3},
    {4, 5, 6},
}

// Iterate
for i, v := range slice {
    fmt.Printf("%d: %d\n", i, v)
}
```

---

## Maps

```go
// Creating maps
var m map[string]int               // nil map
m = make(map[string]int)           // empty map
m = map[string]int{                // map literal
    "Alice": 25,
    "Bob":   30,
}

// Adding/updating elements
m["Charlie"] = 35
m["Alice"] = 26

// Accessing elements
age := m["Alice"]

// Check if key exists
age, ok := m["Alice"]
if ok {
    fmt.Println("Alice's age:", age)
}

// Delete element
delete(m, "Bob")

// Iterate over map
for key, value := range m {
    fmt.Printf("%s: %d\n", key, value)
}

// Map length
size := len(m)

// Nested maps
nested := map[string]map[string]int{
    "group1": {
        "Alice": 25,
        "Bob":   30,
    },
    "group2": {
        "Charlie": 35,
    },
}
```

---

## Control Flow

### If-Else

```go
age := 18

if age < 13 {
    fmt.Println("Child")
} else if age < 20 {
    fmt.Println("Teenager")
} else {
    fmt.Println("Adult")
}

// If with initialization
if num := 42; num > 0 {
    fmt.Println("Positive")
}

// Error checking pattern
if err := someFunction(); err != nil {
    fmt.Println("Error:", err)
}
```

### Switch

```go
// Basic switch
day := 3
switch day {
case 1:
    fmt.Println("Monday")
case 2:
    fmt.Println("Tuesday")
case 3:
    fmt.Println("Wednesday")
default:
    fmt.Println("Other day")
}

// Multiple cases
switch day {
case 1, 2, 3, 4, 5:
    fmt.Println("Weekday")
case 6, 7:
    fmt.Println("Weekend")
}

// Switch with condition
num := 42
switch {
case num < 0:
    fmt.Println("Negative")
case num == 0:
    fmt.Println("Zero")
case num > 0:
    fmt.Println("Positive")
}

// Type switch
var i interface{} = "hello"
switch v := i.(type) {
case string:
    fmt.Println("String:", v)
case int:
    fmt.Println("Int:", v)
default:
    fmt.Println("Unknown type")
}
```

### Loops

```go
// For loop (only loop in Go)
for i := 0; i < 5; i++ {
    fmt.Println(i)
}

// While-style loop
count := 0
for count < 5 {
    fmt.Println(count)
    count++
}

// Infinite loop
for {
    fmt.Println("Forever")
    break  // Exit loop
}

// Range over slice
numbers := []int{1, 2, 3, 4, 5}
for i, v := range numbers {
    fmt.Printf("%d: %d\n", i, v)
}

// Range over map
m := map[string]int{"a": 1, "b": 2}
for key, value := range m {
    fmt.Printf("%s: %d\n", key, value)
}

// Ignore index/value with _
for _, v := range numbers {
    fmt.Println(v)
}

// Break and continue
for i := 0; i < 10; i++ {
    if i == 5 {
        continue
    }
    if i == 8 {
        break
    }
    fmt.Println(i)
}
```

---

## Functions

### Basic Functions

```go
// Simple function
func greet(name string) {
    fmt.Println("Hello,", name)
}

// Function with return value
func add(a, b int) int {
    return a + b
}

// Multiple parameters of same type
func multiply(a, b, c int) int {
    return a * b * c
}

// Multiple return values
func swap(a, b string) (string, string) {
    return b, a
}

// Named return values
func divide(a, b float64) (result float64, err error) {
    if b == 0 {
        err = fmt.Errorf("division by zero")
        return
    }
    result = a / b
    return  // Naked return
}

// Variadic functions
func sum(numbers ...int) int {
    total := 0
    for _, num := range numbers {
        total += num
    }
    return total
}

// Usage
result := sum(1, 2, 3, 4, 5)
```

### Anonymous Functions and Closures

```go
// Anonymous function
add := func(a, b int) int {
    return a + b
}
result := add(5, 3)

// Immediately invoked function
result := func(a, b int) int {
    return a + b
}(5, 3)

// Closure
func counter() func() int {
    count := 0
    return func() int {
        count++
        return count
    }
}

c := counter()
fmt.Println(c())  // 1
fmt.Println(c())  // 2
fmt.Println(c())  // 3
```

### Defer

```go
// Defer executes at function end
func example() {
    defer fmt.Println("World")
    fmt.Println("Hello")
}
// Output: Hello
//         World

// Multiple defers (LIFO order)
func multiDefer() {
    defer fmt.Println("1")
    defer fmt.Println("2")
    defer fmt.Println("3")
}
// Output: 3, 2, 1

// Common pattern: cleanup
func readFile(filename string) error {
    f, err := os.Open(filename)
    if err != nil {
        return err
    }
    defer f.Close()

    // Work with file
    return nil
}
```

---

## Structs and Methods

### Structs

```go
// Define struct
type Person struct {
    Name string
    Age  int
    Email string
}

// Create struct
p1 := Person{
    Name:  "Alice",
    Age:   30,
    Email: "alice@example.com",
}

// Short form
p2 := Person{"Bob", 25, "bob@example.com"}

// Anonymous struct
person := struct {
    name string
    age  int
}{
    name: "Charlie",
    age:  35,
}

// Accessing fields
fmt.Println(p1.Name)
p1.Age = 31

// Pointer to struct
p := &Person{Name: "Alice", Age: 30}
p.Age = 31  // Automatic dereferencing

// Embedded structs
type Address struct {
    City    string
    Country string
}

type Employee struct {
    Person   // Embedded struct
    Address  // Embedded struct
    Salary   float64
}

emp := Employee{
    Person:  Person{Name: "Alice", Age: 30},
    Address: Address{City: "NYC", Country: "USA"},
    Salary:  100000,
}

// Access embedded fields
fmt.Println(emp.Name)  // From Person
fmt.Println(emp.City)  // From Address
```

### Methods

```go
// Method on struct
type Rectangle struct {
    Width  float64
    Height float64
}

// Value receiver
func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

// Pointer receiver (can modify)
func (r *Rectangle) Scale(factor float64) {
    r.Width *= factor
    r.Height *= factor
}

// Usage
rect := Rectangle{Width: 10, Height: 5}
area := rect.Area()
rect.Scale(2)

// Method on any type
type MyInt int

func (m MyInt) Double() MyInt {
    return m * 2
}

num := MyInt(5)
result := num.Double()  // 10
```

---

## Interfaces

```go
// Define interface
type Shape interface {
    Area() float64
    Perimeter() float64
}

// Implement interface (implicit)
type Circle struct {
    Radius float64
}

func (c Circle) Area() float64 {
    return 3.14159 * c.Radius * c.Radius
}

func (c Circle) Perimeter() float64 {
    return 2 * 3.14159 * c.Radius
}

type Rectangle struct {
    Width, Height float64
}

func (r Rectangle) Area() float64 {
    return r.Width * r.Height
}

func (r Rectangle) Perimeter() float64 {
    return 2 * (r.Width + r.Height)
}

// Use interface
func printArea(s Shape) {
    fmt.Printf("Area: %.2f\n", s.Area())
}

// Usage
c := Circle{Radius: 5}
r := Rectangle{Width: 10, Height: 5}

printArea(c)
printArea(r)

// Empty interface (any type)
func printAnything(v interface{}) {
    fmt.Println(v)
}

printAnything(42)
printAnything("hello")
printAnything(true)

// Type assertion
var i interface{} = "hello"
s := i.(string)
s, ok := i.(string)  // Safe type assertion

// Type switch
switch v := i.(type) {
case string:
    fmt.Println("String:", v)
case int:
    fmt.Println("Int:", v)
default:
    fmt.Println("Unknown")
}
```

---

## Concurrency

### Goroutines

```go
// Start goroutine
go func() {
    fmt.Println("Hello from goroutine")
}()

// Multiple goroutines
for i := 0; i < 5; i++ {
    go func(n int) {
        fmt.Println("Goroutine", n)
    }(i)
}

// Wait for goroutines
import "sync"

var wg sync.WaitGroup

for i := 0; i < 5; i++ {
    wg.Add(1)
    go func(n int) {
        defer wg.Done()
        fmt.Println("Worker", n)
    }(i)
}

wg.Wait()
```

### Channels

```go
// Create channel
ch := make(chan int)

// Buffered channel
ch := make(chan int, 5)

// Send to channel
go func() {
    ch <- 42
}()

// Receive from channel
value := <-ch

// Close channel
close(ch)

// Range over channel
go func() {
    for i := 0; i < 5; i++ {
        ch <- i
    }
    close(ch)
}()

for value := range ch {
    fmt.Println(value)
}

// Select statement
ch1 := make(chan string)
ch2 := make(chan string)

go func() {
    ch1 <- "from ch1"
}()

go func() {
    ch2 <- "from ch2"
}()

select {
case msg1 := <-ch1:
    fmt.Println(msg1)
case msg2 := <-ch2:
    fmt.Println(msg2)
case <-time.After(1 * time.Second):
    fmt.Println("timeout")
}
```

### Sync Package

```go
import "sync"

// Mutex
var (
    mu    sync.Mutex
    count int
)

func increment() {
    mu.Lock()
    defer mu.Unlock()
    count++
}

// RWMutex (multiple readers, single writer)
var (
    rwMu sync.RWMutex
    data map[string]int
)

func read(key string) int {
    rwMu.RLock()
    defer rwMu.RUnlock()
    return data[key]
}

func write(key string, value int) {
    rwMu.Lock()
    defer rwMu.Unlock()
    data[key] = value
}

// Once (execute only once)
var once sync.Once

func initialize() {
    once.Do(func() {
        fmt.Println("Initialized")
    })
}
```

---

## Error Handling

```go
import "errors"
import "fmt"

// Return error
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, errors.New("division by zero")
    }
    return a / b, nil
}

// Formatted error
func validateAge(age int) error {
    if age < 0 {
        return fmt.Errorf("invalid age: %d", age)
    }
    return nil
}

// Custom error type
type ValidationError struct {
    Field string
    Value interface{}
}

func (e *ValidationError) Error() string {
    return fmt.Sprintf("validation error: %s = %v", e.Field, e.Value)
}

// Error handling pattern
result, err := divide(10, 0)
if err != nil {
    fmt.Println("Error:", err)
    return
}
fmt.Println("Result:", result)

// Panic and recover
func riskyOperation() {
    defer func() {
        if r := recover(); r != nil {
            fmt.Println("Recovered from panic:", r)
        }
    }()

    panic("something went wrong")
}
```

---

## Packages and Imports

```go
// Package declaration
package main

// Import single package
import "fmt"

// Import multiple packages
import (
    "fmt"
    "math"
    "strings"
)

// Aliased import
import f "fmt"
f.Println("Hello")

// Blank import (side effects)
import _ "database/sql/driver"

// Creating a package
// mypackage/mypackage.go
package mypackage

// Exported (capitalized)
func PublicFunction() {
    fmt.Println("Public")
}

// Not exported (lowercase)
func privateFunction() {
    fmt.Println("Private")
}

// Using the package
import "myproject/mypackage"

mypackage.PublicFunction()
```

---

## File I/O

```go
import (
    "bufio"
    "fmt"
    "io/ioutil"
    "os"
)

// Read entire file
data, err := ioutil.ReadFile("file.txt")
if err != nil {
    panic(err)
}
fmt.Println(string(data))

// Write file
err = ioutil.WriteFile("output.txt", []byte("Hello"), 0644)

// Open file
file, err := os.Open("file.txt")
if err != nil {
    panic(err)
}
defer file.Close()

// Read line by line
scanner := bufio.NewScanner(file)
for scanner.Scan() {
    fmt.Println(scanner.Text())
}

// Write to file
file, err := os.Create("output.txt")
if err != nil {
    panic(err)
}
defer file.Close()

writer := bufio.NewWriter(file)
writer.WriteString("Hello, World!\n")
writer.Flush()
```

---

## Common Patterns

### Singleton

```go
import "sync"

type singleton struct {
    data string
}

var (
    instance *singleton
    once     sync.Once
)

func GetInstance() *singleton {
    once.Do(func() {
        instance = &singleton{data: "singleton"}
    })
    return instance
}
```

### Factory Pattern

```go
type Animal interface {
    Speak() string
}

type Dog struct{}
func (d Dog) Speak() string { return "Woof!" }

type Cat struct{}
func (c Cat) Speak() string { return "Meow!" }

func NewAnimal(animalType string) Animal {
    switch animalType {
    case "dog":
        return Dog{}
    case "cat":
        return Cat{}
    default:
        return nil
    }
}
```

### Builder Pattern

```go
type User struct {
    firstName string
    lastName  string
    age       int
    email     string
}

type UserBuilder struct {
    user User
}

func NewUserBuilder() *UserBuilder {
    return &UserBuilder{}
}

func (b *UserBuilder) FirstName(name string) *UserBuilder {
    b.user.firstName = name
    return b
}

func (b *UserBuilder) LastName(name string) *UserBuilder {
    b.user.lastName = name
    return b
}

func (b *UserBuilder) Age(age int) *UserBuilder {
    b.user.age = age
    return b
}

func (b *UserBuilder) Email(email string) *UserBuilder {
    b.user.email = email
    return b
}

func (b *UserBuilder) Build() User {
    return b.user
}

// Usage
user := NewUserBuilder().
    FirstName("Alice").
    LastName("Smith").
    Age(30).
    Email("alice@example.com").
    Build()
```

---

## Testing

```go
// main.go
package main

func Add(a, b int) int {
    return a + b
}

// main_test.go
package main

import "testing"

func TestAdd(t *testing.T) {
    result := Add(2, 3)
    expected := 5

    if result != expected {
        t.Errorf("Add(2, 3) = %d; want %d", result, expected)
    }
}

func TestAddNegative(t *testing.T) {
    result := Add(-1, -1)
    expected := -2

    if result != expected {
        t.Errorf("Add(-1, -1) = %d; want %d", result, expected)
    }
}

// Table-driven tests
func TestAddTable(t *testing.T) {
    tests := []struct {
        a, b, expected int
    }{
        {1, 2, 3},
        {0, 0, 0},
        {-1, 1, 0},
        {10, 20, 30},
    }

    for _, tt := range tests {
        result := Add(tt.a, tt.b)
        if result != tt.expected {
            t.Errorf("Add(%d, %d) = %d; want %d",
                tt.a, tt.b, result, tt.expected)
        }
    }
}

// Run tests: go test
// Run with coverage: go test -cover
```

---

## Best Practices

1. **Use gofmt** - Format code automatically
   ```bash
   gofmt -w .
   ```

2. **Use golint** - Check code style
   ```bash
   golint ./...
   ```

3. **Error handling** - Always check errors
   ```go
   if err != nil {
       return err
   }
   ```

4. **Use interfaces** - Program to interfaces, not implementations

5. **Prefer composition** over inheritance

6. **Keep functions small** - Single responsibility

7. **Use meaningful names** - Clear and descriptive

8. **Document exported items** - Comments for public API
   ```go
   // Add returns the sum of two integers.
   func Add(a, b int) int {
       return a + b
   }
   ```

9. **Use context** for cancellation and timeouts
   ```go
   ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
   defer cancel()
   ```

10. **Avoid global state** - Pass dependencies explicitly

---

## Common Libraries

- **gorilla/mux**: HTTP router
- **gin**: Web framework
- **gorm**: ORM
- **viper**: Configuration
- **cobra**: CLI applications
- **logrus**: Logging
- **testify**: Testing toolkit
- **zap**: Fast logging
- **grpc**: RPC framework
- **redis**: Redis client

---

## Go Modules

```bash
# Initialize module
go mod init github.com/username/project

# Add dependency
go get github.com/gin-gonic/gin

# Update dependencies
go get -u

# Tidy dependencies
go mod tidy

# Vendor dependencies
go mod vendor
```

---

## Useful Commands

```bash
# Run program
go run main.go

# Build executable
go build

# Install binary
go install

# Format code
go fmt ./...

# Run tests
go test ./...

# Get dependencies
go get package

# Show documentation
go doc fmt.Println
```
