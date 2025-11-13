# Rust Programming

## Overview

Rust is a systems programming language focused on safety, speed, and concurrency. It achieves memory safety without garbage collection through its ownership system.

**Key Features:**
- Memory safety without garbage collection
- Zero-cost abstractions
- Ownership and borrowing system
- Guaranteed thread safety
- Pattern matching
- Type inference
- Powerful macro system
- Excellent tooling (cargo, rustfmt, clippy)

---

## Basic Syntax

### Variables and Data Types

```rust
fn main() {
    // Immutable by default
    let x = 5;
    // x = 6; // Error! Cannot mutate immutable variable

    // Mutable variable
    let mut y = 5;
    y = 6; // OK

    // Constants (must have type annotation)
    const MAX_POINTS: u32 = 100_000;

    // Shadowing (redefining variable)
    let x = 5;
    let x = x + 1;
    let x = x * 2; // x is now 12

    // Type annotation
    let guess: u32 = "42".parse().expect("Not a number!");

    // Scalar types
    let integer: i32 = 42;
    let float: f64 = 3.14;
    let boolean: bool = true;
    let character: char = 'A';

    // Integer types: i8, i16, i32, i64, i128, isize
    // Unsigned: u8, u16, u32, u64, u128, usize
    let signed: i8 = -127;
    let unsigned: u8 = 255;

    // Number literals
    let decimal = 98_222;
    let hex = 0xff;
    let octal = 0o77;
    let binary = 0b1111_0000;
    let byte = b'A'; // u8 only
}
```

### Strings

```rust
fn main() {
    // String slice (immutable, fixed size)
    let s1: &str = "Hello";

    // String (mutable, growable)
    let mut s2 = String::from("Hello");
    s2.push_str(", World!");

    // String operations
    let len = s2.len();
    let is_empty = s2.is_empty();
    let contains = s2.contains("World");

    // String concatenation
    let s3 = String::from("Hello");
    let s4 = String::from(" World");
    let s5 = s3 + &s4; // s3 is moved, can't use it anymore

    // Format macro (doesn't take ownership)
    let s6 = format!("{} {}", s1, s4);

    // String slicing
    let hello = &s2[0..5];
    let world = &s2[7..12];

    // Iterate over chars
    for c in "Hello".chars() {
        println!("{}", c);
    }

    // Iterate over bytes
    for b in "Hello".bytes() {
        println!("{}", b);
    }

    // String to number
    let num: i32 = "42".parse().unwrap();
}
```

---

## Ownership and Borrowing

### Ownership Rules

1. Each value in Rust has a variable called its owner
2. There can only be one owner at a time
3. When the owner goes out of scope, the value is dropped

```rust
fn main() {
    // Move (ownership transfer)
    let s1 = String::from("hello");
    let s2 = s1; // s1 is no longer valid
    // println!("{}", s1); // Error!
    println!("{}", s2); // OK

    // Clone (deep copy)
    let s3 = String::from("hello");
    let s4 = s3.clone();
    println!("{} {}", s3, s4); // Both valid

    // Copy trait (stack-only data)
    let x = 5;
    let y = x; // x is still valid (Copy trait)
    println!("{} {}", x, y);
}
```

### References and Borrowing

```rust
fn main() {
    // Immutable reference (borrowing)
    let s1 = String::from("hello");
    let len = calculate_length(&s1); // Borrow
    println!("{} has length {}", s1, len); // s1 still valid

    // Mutable reference
    let mut s = String::from("hello");
    change(&mut s);
    println!("{}", s); // "hello, world"

    // Rules:
    // 1. Multiple immutable references OR one mutable reference
    // 2. References must always be valid

    let r1 = &s; // OK
    let r2 = &s; // OK
    // let r3 = &mut s; // Error! Can't have mutable while immutable exists

    println!("{} {}", r1, r2);
    // r1 and r2 no longer used after this

    let r3 = &mut s; // OK now
}

fn calculate_length(s: &String) -> usize {
    s.len()
}

fn change(s: &mut String) {
    s.push_str(", world");
}
```

### Lifetimes

```rust
// Lifetime annotations
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

// Struct with lifetime
struct ImportantExcerpt<'a> {
    part: &'a str,
}

impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
}

fn main() {
    let string1 = String::from("long string");
    let result;
    {
        let string2 = String::from("short");
        result = longest(string1.as_str(), string2.as_str());
        println!("Longest: {}", result);
    }
    // result not valid here (string2 dropped)
}
```

---

## Data Structures

### Arrays and Vectors

```rust
fn main() {
    // Array (fixed size)
    let arr: [i32; 5] = [1, 2, 3, 4, 5];
    let arr2 = [3; 5]; // [3, 3, 3, 3, 3]
    let first = arr[0];
    let len = arr.len();

    // Vector (dynamic array)
    let mut vec = Vec::new();
    vec.push(1);
    vec.push(2);
    vec.push(3);

    // Vec macro
    let vec2 = vec![1, 2, 3, 4, 5];

    // Accessing elements
    let third = &vec2[2];
    let third = vec2.get(2); // Returns Option<&T>

    // Iterate
    for i in &vec2 {
        println!("{}", i);
    }

    // Mutable iteration
    let mut vec3 = vec![1, 2, 3];
    for i in &mut vec3 {
        *i += 50;
    }

    // Vector with enum for multiple types
    enum SpreadsheetCell {
        Int(i32),
        Float(f64),
        Text(String),
    }

    let row = vec![
        SpreadsheetCell::Int(3),
        SpreadsheetCell::Float(10.12),
        SpreadsheetCell::Text(String::from("blue")),
    ];
}
```

### HashMap

```rust
use std::collections::HashMap;

fn main() {
    // Create HashMap
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);

    // From vectors
    let teams = vec![String::from("Blue"), String::from("Yellow")];
    let initial_scores = vec![10, 50];
    let scores: HashMap<_, _> = teams.iter().zip(initial_scores.iter()).collect();

    // Accessing values
    let team_name = String::from("Blue");
    let score = scores.get(&team_name); // Returns Option<&V>

    // Iterate
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }

    // Update values
    scores.insert(String::from("Blue"), 25); // Overwrite

    // Only insert if key doesn't exist
    scores.entry(String::from("Blue")).or_insert(50);

    // Update based on old value
    let text = "hello world wonderful world";
    let mut map = HashMap::new();

    for word in text.split_whitespace() {
        let count = map.entry(word).or_insert(0);
        *count += 1;
    }

    println!("{:?}", map);
}
```

---

## Control Flow

### If-Else

```rust
fn main() {
    let number = 6;

    if number % 4 == 0 {
        println!("divisible by 4");
    } else if number % 3 == 0 {
        println!("divisible by 3");
    } else {
        println!("not divisible by 4 or 3");
    }

    // If in let statement
    let condition = true;
    let number = if condition { 5 } else { 6 };
}
```

### Loops

```rust
fn main() {
    // Loop (infinite)
    let mut count = 0;
    let result = loop {
        count += 1;
        if count == 10 {
            break count * 2; // Return value
        }
    };

    // While loop
    let mut number = 3;
    while number != 0 {
        println!("{}!", number);
        number -= 1;
    }

    // For loop
    let arr = [10, 20, 30, 40, 50];
    for element in arr.iter() {
        println!("{}", element);
    }

    // Range
    for number in 1..4 {
        println!("{}", number); // 1, 2, 3
    }

    // Reverse range
    for number in (1..4).rev() {
        println!("{}", number); // 3, 2, 1
    }

    // Enumerate
    for (i, v) in arr.iter().enumerate() {
        println!("{}: {}", i, v);
    }
}
```

### Match

```rust
fn main() {
    // Basic match
    let number = 3;
    match number {
        1 => println!("One"),
        2 => println!("Two"),
        3 => println!("Three"),
        _ => println!("Other"), // Default case
    }

    // Match with return value
    let result = match number {
        1 => "one",
        2 => "two",
        3 => "three",
        _ => "other",
    };

    // Match ranges
    match number {
        1..=5 => println!("1 through 5"),
        _ => println!("something else"),
    }

    // Match Option
    let some_value: Option<i32> = Some(3);
    match some_value {
        Some(i) => println!("Got {}", i),
        None => println!("Got nothing"),
    }

    // if let (concise match)
    if let Some(i) = some_value {
        println!("{}", i);
    }

    // Match guards
    let num = Some(4);
    match num {
        Some(x) if x < 5 => println!("less than five: {}", x),
        Some(x) => println!("{}", x),
        None => (),
    }
}
```

---

## Structs and Enums

### Structs

```rust
// Define struct
struct User {
    username: String,
    email: String,
    sign_in_count: u64,
    active: bool,
}

// Tuple struct
struct Color(i32, i32, i32);
struct Point(i32, i32, i32);

// Unit struct (no fields)
struct AlwaysEqual;

impl User {
    // Associated function (constructor)
    fn new(username: String, email: String) -> User {
        User {
            username,
            email,
            sign_in_count: 1,
            active: true,
        }
    }

    // Method
    fn is_active(&self) -> bool {
        self.active
    }

    // Mutable method
    fn deactivate(&mut self) {
        self.active = false;
    }
}

fn main() {
    // Create instance
    let mut user1 = User {
        email: String::from("user@example.com"),
        username: String::from("user123"),
        active: true,
        sign_in_count: 1,
    };

    user1.email = String::from("newemail@example.com");

    // Struct update syntax
    let user2 = User {
        email: String::from("another@example.com"),
        ..user1 // Rest from user1
    };

    // Tuple struct
    let black = Color(0, 0, 0);
    let origin = Point(0, 0, 0);
}
```

### Enums

```rust
// Define enum
enum IpAddr {
    V4(u8, u8, u8, u8),
    V6(String),
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn call(&self) {
        match self {
            Message::Quit => println!("Quit"),
            Message::Move { x, y } => println!("Move to {}, {}", x, y),
            Message::Write(text) => println!("Write: {}", text),
            Message::ChangeColor(r, g, b) => println!("Color: {}, {}, {}", r, g, b),
        }
    }
}

fn main() {
    let home = IpAddr::V4(127, 0, 0, 1);
    let loopback = IpAddr::V6(String::from("::1"));

    let msg = Message::Write(String::from("hello"));
    msg.call();
}
```

### Option and Result

```rust
fn main() {
    // Option<T> - value or nothing
    let some_number: Option<i32> = Some(5);
    let no_number: Option<i32> = None;

    // Match on Option
    match some_number {
        Some(i) => println!("{}", i),
        None => println!("nothing"),
    }

    // Unwrap (panics if None)
    let x = Some(5);
    let y = x.unwrap();

    // Unwrap with default
    let z = no_number.unwrap_or(0);

    // Result<T, E> - success or error
    use std::fs::File;
    use std::io::ErrorKind;

    let f = File::open("hello.txt");
    let f = match f {
        Ok(file) => file,
        Err(error) => match error.kind() {
            ErrorKind::NotFound => match File::create("hello.txt") {
                Ok(fc) => fc,
                Err(e) => panic!("Problem creating file: {:?}", e),
            },
            other_error => panic!("Problem opening file: {:?}", other_error),
        },
    };

    // Propagating errors with ?
    fn read_username() -> Result<String, std::io::Error> {
        let mut f = File::open("hello.txt")?;
        let mut s = String::new();
        use std::io::Read;
        f.read_to_string(&mut s)?;
        Ok(s)
    }
}
```

---

## Traits

```rust
// Define trait
trait Summary {
    fn summarize(&self) -> String;

    // Default implementation
    fn default_summary(&self) -> String {
        String::from("(Read more...)")
    }
}

// Implement trait
struct NewsArticle {
    headline: String,
    location: String,
    author: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

struct Tweet {
    username: String,
    content: String,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}

// Trait as parameter
fn notify(item: &impl Summary) {
    println!("Breaking news! {}", item.summarize());
}

// Trait bound syntax
fn notify2<T: Summary>(item: &T) {
    println!("{}", item.summarize());
}

// Multiple traits
fn notify3<T: Summary + Display>(item: &T) {
    // ...
}

// Where clause
fn some_function<T, U>(t: &T, u: &U) -> i32
where
    T: Display + Clone,
    U: Clone + Debug,
{
    // ...
    0
}

// Return trait
fn returns_summarizable() -> impl Summary {
    Tweet {
        username: String::from("user"),
        content: String::from("content"),
    }
}

use std::fmt::Display;
use std::fmt::Debug;

fn main() {
    let tweet = Tweet {
        username: String::from("user"),
        content: String::from("Hello, world!"),
    };

    println!("{}", tweet.summarize());
}
```

---

## Error Handling

```rust
use std::fs::File;
use std::io::{self, Read};

// Propagating errors
fn read_username_from_file() -> Result<String, io::Error> {
    let mut f = File::open("username.txt")?;
    let mut s = String::new();
    f.read_to_string(&mut s)?;
    Ok(s)
}

// Custom error types
use std::fmt;

#[derive(Debug)]
enum CustomError {
    IoError(io::Error),
    ParseError,
}

impl fmt::Display for CustomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CustomError::IoError(e) => write!(f, "IO error: {}", e),
            CustomError::ParseError => write!(f, "Parse error"),
        }
    }
}

impl From<io::Error> for CustomError {
    fn from(err: io::Error) -> CustomError {
        CustomError::IoError(err)
    }
}

// Panic
fn will_panic() {
    panic!("crash and burn");
}

// Assert
fn check_value(x: i32) {
    assert!(x > 0, "x must be positive");
    assert_eq!(x, 5);
    assert_ne!(x, 0);
}

fn main() {
    // Result with match
    match read_username_from_file() {
        Ok(username) => println!("Username: {}", username),
        Err(e) => println!("Error: {}", e),
    }

    // Unwrap
    let f = File::open("hello.txt").unwrap();

    // Expect (with custom message)
    let f = File::open("hello.txt").expect("Failed to open file");
}
```

---

## Generics

```rust
// Generic function
fn largest<T: PartialOrd>(list: &[T]) -> &T {
    let mut largest = &list[0];
    for item in list {
        if item > largest {
            largest = item;
        }
    }
    largest
}

// Generic struct
struct Point<T> {
    x: T,
    y: T,
}

impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}

// Implement for specific type
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}

// Multiple generic types
struct Point2<T, U> {
    x: T,
    y: U,
}

// Generic enum
enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

fn main() {
    let numbers = vec![34, 50, 25, 100, 65];
    let result = largest(&numbers);

    let integer = Point { x: 5, y: 10 };
    let float = Point { x: 1.0, y: 4.0 };
    let mixed = Point2 { x: 5, y: 4.0 };
}
```

---

## Concurrency

### Threads

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // Spawn thread
    let handle = thread::spawn(|| {
        for i in 1..10 {
            println!("spawned thread: {}", i);
            thread::sleep(Duration::from_millis(1));
        }
    });

    // Wait for thread
    handle.join().unwrap();

    // Move closure
    let v = vec![1, 2, 3];
    let handle = thread::spawn(move || {
        println!("vector: {:?}", v);
    });

    handle.join().unwrap();
}
```

### Channels

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    // Create channel
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let val = String::from("hi");
        tx.send(val).unwrap();
    });

    let received = rx.recv().unwrap();
    println!("Got: {}", received);

    // Multiple producers
    let (tx, rx) = mpsc::channel();
    let tx1 = tx.clone();

    thread::spawn(move || {
        tx.send(String::from("hi from thread 1")).unwrap();
    });

    thread::spawn(move || {
        tx1.send(String::from("hi from thread 2")).unwrap();
    });

    for received in rx {
        println!("Got: {}", received);
    }
}
```

### Shared State

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    // Mutex for mutual exclusion
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

---

## Common Patterns

### Builder Pattern

```rust
#[derive(Default)]
struct User {
    username: String,
    email: String,
    age: Option<u32>,
}

impl User {
    fn builder() -> UserBuilder {
        UserBuilder::default()
    }
}

#[derive(Default)]
struct UserBuilder {
    username: String,
    email: String,
    age: Option<u32>,
}

impl UserBuilder {
    fn username(mut self, username: &str) -> Self {
        self.username = username.to_string();
        self
    }

    fn email(mut self, email: &str) -> Self {
        self.email = email.to_string();
        self
    }

    fn age(mut self, age: u32) -> Self {
        self.age = Some(age);
        self
    }

    fn build(self) -> User {
        User {
            username: self.username,
            email: self.email,
            age: self.age,
        }
    }
}

fn main() {
    let user = User::builder()
        .username("alice")
        .email("alice@example.com")
        .age(30)
        .build();
}
```

### Newtype Pattern

```rust
// Wrap existing type
struct Wrapper(Vec<String>);

impl Wrapper {
    fn new() -> Self {
        Wrapper(Vec::new())
    }

    fn push(&mut self, s: String) {
        self.0.push(s);
    }
}
```

---

## Testing

```rust
// Unit tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn it_adds() {
        assert_eq!(add(2, 2), 4);
    }

    #[test]
    #[should_panic]
    fn it_panics() {
        panic!("panic!");
    }

    #[test]
    fn it_returns_result() -> Result<(), String> {
        if 2 + 2 == 4 {
            Ok(())
        } else {
            Err(String::from("two plus two does not equal four"))
        }
    }
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

---

## Cargo Commands

```bash
# Create new project
cargo new project_name
cargo new --lib library_name

# Build project
cargo build
cargo build --release

# Run project
cargo run

# Check code
cargo check

# Run tests
cargo test

# Generate documentation
cargo doc --open

# Update dependencies
cargo update

# Format code
cargo fmt

# Lint code
cargo clippy
```

---

## Best Practices

1. **Use ownership properly** - Avoid unnecessary clones
2. **Handle errors with Result** - Don't unwrap in production
3. **Use iterators** - More efficient and idiomatic
4. **Prefer `&str` over `String`** for function parameters
5. **Use `Option` instead of null**
6. **Implement `Debug` trait** for custom types
7. **Use pattern matching** instead of if chains
8. **Follow naming conventions** - snake_case for variables/functions
9. **Write tests** - `cargo test`
10. **Use clippy** - `cargo clippy` for linting

---

## Common Libraries

- **serde**: Serialization/deserialization
- **tokio**: Async runtime
- **reqwest**: HTTP client
- **actix-web**: Web framework
- **diesel**: ORM
- **clap**: CLI argument parsing
- **log**: Logging facade
- **anyhow**: Error handling
- **thiserror**: Custom error types
