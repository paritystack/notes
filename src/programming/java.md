# Java Programming

## Overview

Java is a high-level, class-based, object-oriented programming language designed to have minimal implementation dependencies. It follows the "write once, run anywhere" (WORA) principle.

**Key Features:**
- Platform independent (runs on JVM)
- Object-oriented programming
- Automatic memory management (Garbage Collection)
- Strong type system
- Rich standard library
- Multi-threading support

---

## Basic Syntax

### Variables and Data Types

```java
// Primitive types
byte b = 127;           // 8-bit
short s = 32767;        // 16-bit
int i = 2147483647;     // 32-bit
long l = 9223372036854775807L;  // 64-bit
float f = 3.14f;        // 32-bit floating point
double d = 3.14159;     // 64-bit floating point
boolean bool = true;    // true or false
char c = 'A';           // 16-bit Unicode

// Reference types
String str = "Hello, World!";
Integer num = 42;  // Wrapper class

// Type conversion
int x = (int) 3.14;  // Explicit casting
double y = 10;       // Implicit casting

// Constants
final double PI = 3.14159;
final int MAX_SIZE = 100;
```

### String Operations

```java
// String creation
String s1 = "Hello";
String s2 = new String("World");

// String methods
int length = s1.length();
char ch = s1.charAt(0);
String sub = s1.substring(0, 3);
String upper = s1.toUpperCase();
String lower = s1.toLowerCase();
boolean startsWith = s1.startsWith("He");
boolean contains = s1.contains("ll");

// String comparison
boolean equals = s1.equals(s2);
boolean equalsIgnoreCase = s1.equalsIgnoreCase(s2);
int compare = s1.compareTo(s2);

// String concatenation
String full = s1 + " " + s2;
String joined = String.join(", ", "a", "b", "c");

// String formatting
String formatted = String.format("Name: %s, Age: %d", "Alice", 30);

// StringBuilder (mutable)
StringBuilder sb = new StringBuilder();
sb.append("Hello");
sb.append(" World");
String result = sb.toString();
```

---

## Arrays and Collections

### Arrays

```java
// Array declaration
int[] numbers = new int[5];
int[] nums = {1, 2, 3, 4, 5};
String[] names = {"Alice", "Bob", "Charlie"};

// Accessing elements
int first = nums[0];
nums[2] = 10;

// Array length
int length = nums.length;

// Multi-dimensional arrays
int[][] matrix = new int[3][3];
int[][] grid = {{1, 2}, {3, 4}, {5, 6}};

// Arrays utility class
import java.util.Arrays;

Arrays.sort(nums);                    // Sort array
int index = Arrays.binarySearch(nums, 5);  // Binary search
int[] copy = Arrays.copyOf(nums, nums.length);  // Copy
boolean equal = Arrays.equals(nums, copy);  // Compare
String str = Arrays.toString(nums);   // Convert to string
```

### ArrayList

```java
import java.util.ArrayList;

// Creating ArrayList
ArrayList<String> list = new ArrayList<>();
ArrayList<Integer> numbers = new ArrayList<>(Arrays.asList(1, 2, 3));

// Adding elements
list.add("Apple");
list.add(0, "Banana");  // Add at index
list.addAll(Arrays.asList("Cherry", "Date"));

// Accessing elements
String first = list.get(0);
list.set(1, "Blueberry");

// Removing elements
list.remove(0);
list.remove("Apple");
list.clear();

// Operations
int size = list.size();
boolean empty = list.isEmpty();
boolean contains = list.contains("Apple");
int index = list.indexOf("Apple");

// Iteration
for (String item : list) {
    System.out.println(item);
}

list.forEach(item -> System.out.println(item));
```

### HashMap

```java
import java.util.HashMap;
import java.util.Map;

// Creating HashMap
HashMap<String, Integer> map = new HashMap<>();

// Adding elements
map.put("Alice", 25);
map.put("Bob", 30);
map.putIfAbsent("Charlie", 35);

// Accessing elements
int age = map.get("Alice");
int defaultAge = map.getOrDefault("David", 0);

// Removing elements
map.remove("Bob");

// Operations
int size = map.size();
boolean empty = map.isEmpty();
boolean hasKey = map.containsKey("Alice");
boolean hasValue = map.containsValue(25);

// Iteration
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + ": " + entry.getValue());
}

map.forEach((key, value) ->
    System.out.println(key + ": " + value));
```

---

## Control Flow

### If-Else

```java
int age = 18;

if (age < 13) {
    System.out.println("Child");
} else if (age < 20) {
    System.out.println("Teenager");
} else {
    System.out.println("Adult");
}

// Ternary operator
String status = (age >= 18) ? "Adult" : "Minor";
```

### Switch

```java
// Traditional switch
int day = 3;
switch (day) {
    case 1:
        System.out.println("Monday");
        break;
    case 2:
        System.out.println("Tuesday");
        break;
    default:
        System.out.println("Other day");
}

// Switch expression (Java 14+)
String dayName = switch (day) {
    case 1 -> "Monday";
    case 2 -> "Tuesday";
    case 3 -> "Wednesday";
    default -> "Other day";
};
```

### Loops

```java
// For loop
for (int i = 0; i < 5; i++) {
    System.out.println(i);
}

// Enhanced for loop
int[] numbers = {1, 2, 3, 4, 5};
for (int num : numbers) {
    System.out.println(num);
}

// While loop
int count = 0;
while (count < 5) {
    System.out.println(count);
    count++;
}

// Do-while loop
int i = 0;
do {
    System.out.println(i);
    i++;
} while (i < 5);

// Break and continue
for (int j = 0; j < 10; j++) {
    if (j == 5) continue;  // Skip 5
    if (j == 8) break;     // Stop at 8
    System.out.println(j);
}
```

---

## Object-Oriented Programming

### Classes and Objects

```java
public class Person {
    // Fields (instance variables)
    private String name;
    private int age;

    // Static field (class variable)
    private static int count = 0;

    // Constructor
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
        count++;
    }

    // Default constructor
    public Person() {
        this("Unknown", 0);
    }

    // Getters and setters
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        if (age >= 0) {
            this.age = age;
        }
    }

    // Instance method
    public void greet() {
        System.out.println("Hello, I'm " + name);
    }

    // Static method
    public static int getCount() {
        return count;
    }

    // toString method
    @Override
    public String toString() {
        return "Person{name='" + name + "', age=" + age + "}";
    }
}

// Usage
Person person = new Person("Alice", 30);
person.greet();
System.out.println(person.toString());
```

### Inheritance

```java
// Base class
public class Animal {
    protected String name;

    public Animal(String name) {
        this.name = name;
    }

    public void speak() {
        System.out.println(name + " makes a sound");
    }
}

// Derived class
public class Dog extends Animal {
    private String breed;

    public Dog(String name, String breed) {
        super(name);  // Call parent constructor
        this.breed = breed;
    }

    @Override
    public void speak() {
        System.out.println(name + " barks");
    }

    public void fetch() {
        System.out.println(name + " is fetching");
    }
}

// Usage
Dog dog = new Dog("Buddy", "Golden Retriever");
dog.speak();   // "Buddy barks"
dog.fetch();   // "Buddy is fetching"
```

### Interfaces

```java
// Interface definition
public interface Drawable {
    void draw();  // Abstract method

    // Default method (Java 8+)
    default void display() {
        System.out.println("Displaying...");
    }

    // Static method (Java 8+)
    static void info() {
        System.out.println("Drawable interface");
    }
}

// Implementation
public class Circle implements Drawable {
    private double radius;

    public Circle(double radius) {
        this.radius = radius;
    }

    @Override
    public void draw() {
        System.out.println("Drawing circle with radius " + radius);
    }
}

// Multiple interfaces
public class Square implements Drawable, Comparable<Square> {
    private double side;

    public Square(double side) {
        this.side = side;
    }

    @Override
    public void draw() {
        System.out.println("Drawing square with side " + side);
    }

    @Override
    public int compareTo(Square other) {
        return Double.compare(this.side, other.side);
    }
}
```

### Abstract Classes

```java
public abstract class Shape {
    protected String color;

    public Shape(String color) {
        this.color = color;
    }

    // Abstract method
    public abstract double area();

    // Concrete method
    public void setColor(String color) {
        this.color = color;
    }

    public String getColor() {
        return color;
    }
}

public class Rectangle extends Shape {
    private double width;
    private double height;

    public Rectangle(String color, double width, double height) {
        super(color);
        this.width = width;
        this.height = height;
    }

    @Override
    public double area() {
        return width * height;
    }
}
```

---

## Exception Handling

```java
// Try-catch
try {
    int result = 10 / 0;
} catch (ArithmeticException e) {
    System.out.println("Cannot divide by zero!");
}

// Multiple catch blocks
try {
    int[] arr = new int[5];
    arr[10] = 50;
} catch (ArrayIndexOutOfBoundsException e) {
    System.out.println("Array index out of bounds");
} catch (Exception e) {
    System.out.println("General exception: " + e.getMessage());
}

// Finally block
try {
    // Code that may throw exception
} catch (Exception e) {
    e.printStackTrace();
} finally {
    System.out.println("This always executes");
}

// Try-with-resources (Java 7+)
try (BufferedReader br = new BufferedReader(new FileReader("file.txt"))) {
    String line = br.readLine();
} catch (IOException e) {
    e.printStackTrace();
}

// Throwing exceptions
public void checkAge(int age) throws IllegalArgumentException {
    if (age < 0) {
        throw new IllegalArgumentException("Age cannot be negative");
    }
}

// Custom exception
public class InvalidAgeException extends Exception {
    public InvalidAgeException(String message) {
        super(message);
    }
}
```

---

## Streams and Lambdas (Java 8+)

### Lambda Expressions

```java
// Functional interface
@FunctionalInterface
interface Calculator {
    int calculate(int a, int b);
}

// Lambda expression
Calculator add = (a, b) -> a + b;
Calculator multiply = (a, b) -> a * b;

System.out.println(add.calculate(5, 3));      // 8
System.out.println(multiply.calculate(5, 3)); // 15

// With collections
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
names.forEach(name -> System.out.println(name));

// Method reference
names.forEach(System.out::println);
```

### Streams

```java
import java.util.stream.*;

List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

// Filter
List<Integer> evens = numbers.stream()
    .filter(n -> n % 2 == 0)
    .collect(Collectors.toList());

// Map
List<Integer> squared = numbers.stream()
    .map(n -> n * n)
    .collect(Collectors.toList());

// Reduce
int sum = numbers.stream()
    .reduce(0, (a, b) -> a + b);

// Find
Optional<Integer> first = numbers.stream()
    .filter(n -> n > 5)
    .findFirst();

// Any/All match
boolean anyEven = numbers.stream().anyMatch(n -> n % 2 == 0);
boolean allPositive = numbers.stream().allMatch(n -> n > 0);

// Sorted
List<Integer> sorted = numbers.stream()
    .sorted()
    .collect(Collectors.toList());

// Limit and skip
List<Integer> limited = numbers.stream()
    .limit(5)
    .collect(Collectors.toList());

// Chaining operations
List<String> result = Arrays.asList("apple", "banana", "cherry", "date")
    .stream()
    .filter(s -> s.length() > 5)
    .map(String::toUpperCase)
    .sorted()
    .collect(Collectors.toList());
```

---

## Common Patterns

### Singleton

```java
public class Singleton {
    private static Singleton instance;

    private Singleton() {
        // Private constructor
    }

    public static Singleton getInstance() {
        if (instance == null) {
            instance = new Singleton();
        }
        return instance;
    }
}

// Thread-safe singleton
public class ThreadSafeSingleton {
    private static volatile ThreadSafeSingleton instance;

    private ThreadSafeSingleton() {}

    public static ThreadSafeSingleton getInstance() {
        if (instance == null) {
            synchronized (ThreadSafeSingleton.class) {
                if (instance == null) {
                    instance = new ThreadSafeSingleton();
                }
            }
        }
        return instance;
    }
}
```

### Factory Pattern

```java
interface Animal {
    void speak();
}

class Dog implements Animal {
    public void speak() {
        System.out.println("Woof!");
    }
}

class Cat implements Animal {
    public void speak() {
        System.out.println("Meow!");
    }
}

class AnimalFactory {
    public static Animal createAnimal(String type) {
        if (type.equals("dog")) {
            return new Dog();
        } else if (type.equals("cat")) {
            return new Cat();
        }
        throw new IllegalArgumentException("Unknown animal type");
    }
}

// Usage
Animal animal = AnimalFactory.createAnimal("dog");
animal.speak();
```

### Builder Pattern

```java
public class User {
    private final String firstName;
    private final String lastName;
    private final int age;
    private final String email;

    private User(UserBuilder builder) {
        this.firstName = builder.firstName;
        this.lastName = builder.lastName;
        this.age = builder.age;
        this.email = builder.email;
    }

    public static class UserBuilder {
        private String firstName;
        private String lastName;
        private int age;
        private String email;

        public UserBuilder(String firstName, String lastName) {
            this.firstName = firstName;
            this.lastName = lastName;
        }

        public UserBuilder age(int age) {
            this.age = age;
            return this;
        }

        public UserBuilder email(String email) {
            this.email = email;
            return this;
        }

        public User build() {
            return new User(this);
        }
    }
}

// Usage
User user = new User.UserBuilder("Alice", "Smith")
    .age(30)
    .email("alice@example.com")
    .build();
```

---

## File I/O

```java
import java.io.*;
import java.nio.file.*;

// Reading file
try (BufferedReader br = new BufferedReader(new FileReader("file.txt"))) {
    String line;
    while ((line = br.readLine()) != null) {
        System.out.println(line);
    }
} catch (IOException e) {
    e.printStackTrace();
}

// Writing file
try (BufferedWriter bw = new BufferedWriter(new FileWriter("file.txt"))) {
    bw.write("Hello, World!");
    bw.newLine();
    bw.write("Second line");
} catch (IOException e) {
    e.printStackTrace();
}

// Using Files class (Java 7+)
try {
    // Read all lines
    List<String> lines = Files.readAllLines(Paths.get("file.txt"));

    // Write lines
    Files.write(Paths.get("output.txt"),
                Arrays.asList("Line 1", "Line 2"));

    // Copy file
    Files.copy(Paths.get("source.txt"), Paths.get("dest.txt"));

    // Delete file
    Files.delete(Paths.get("file.txt"));
} catch (IOException e) {
    e.printStackTrace();
}
```

---

## Best Practices

1. **Follow naming conventions**
   - Classes: PascalCase (`MyClass`)
   - Methods/variables: camelCase (`myMethod`)
   - Constants: UPPER_SNAKE_CASE (`MAX_SIZE`)

2. **Use meaningful names**
   ```java
   // Good
   int studentCount = 50;

   // Bad
   int sc = 50;
   ```

3. **Keep methods small** - One responsibility per method

4. **Use StringBuilder** for string concatenation in loops

5. **Close resources** - Use try-with-resources

6. **Handle exceptions properly** - Don't swallow exceptions

7. **Use generics** for type safety

8. **Follow SOLID principles**

9. **Use Optional** to avoid null checks (Java 8+)
   ```java
   Optional<String> optional = Optional.ofNullable(getValue());
   String value = optional.orElse("default");
   ```

10. **Use streams** for collection processing (Java 8+)

---

## Common Libraries/Frameworks

- **Spring Boot**: Application framework
- **Hibernate**: ORM framework
- **JUnit**: Testing framework
- **Maven/Gradle**: Build tools
- **Jackson**: JSON processing
- **Log4j/SLF4J**: Logging
- **Apache Commons**: Utility libraries
