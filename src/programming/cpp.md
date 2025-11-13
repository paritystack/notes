# C++

## Overview
C++ is an extension of C that adds object-oriented features and other enhancements.

## Key Features
- Object-oriented programming
- Generic programming support
- Standard Template Library (STL)
- Low-level memory manipulation
- High performance

## Object Instantiation Patterns

C++ provides multiple ways to create and initialize objects, each with different characteristics regarding memory management, lifetime, and performance.

### 1. **Stack Allocation (Automatic Storage)**

Objects created on the stack have automatic lifetime - they're destroyed when they go out of scope.

```cpp
class MyClass {
public:
    int value;
    MyClass(int v) : value(v) {
        std::cout << "Constructor called: " << value << std::endl;
    }
    ~MyClass() {
        std::cout << "Destructor called: " << value << std::endl;
    }
};

void example() {
    MyClass obj1(10);           // Stack allocation
    MyClass obj2 = MyClass(20); // Also stack allocation
    MyClass obj3{30};           // C++11 uniform initialization

    // All objects destroyed automatically when function exits
}
```

**Advantages:**
- Fast allocation/deallocation
- Automatic cleanup (RAII)
- No memory leaks

**Disadvantages:**
- Limited stack size
- Objects can't outlive their scope

### 2. **Heap Allocation with new/delete**

Objects created on the heap persist until explicitly deleted.

```cpp
// Single object
MyClass* ptr1 = new MyClass(100);  // Allocate on heap
// Use ptr1...
delete ptr1;  // Must manually delete
ptr1 = nullptr;  // Good practice

// Array of objects
MyClass* arr = new MyClass[5];  // Default constructor for each
// Use arr...
delete[] arr;  // Must use delete[] for arrays
arr = nullptr;

// With initialization (C++11)
MyClass* ptr2 = new MyClass{200};
delete ptr2;
```

**Advantages:**
- Objects can outlive their scope
- Larger available memory
- Dynamic sizing

**Disadvantages:**
- Manual memory management
- Risk of memory leaks
- Slower than stack allocation

### 3. **Smart Pointers (Modern C++)**

Smart pointers provide automatic memory management for heap-allocated objects.

```cpp
#include <memory>

// std::unique_ptr - exclusive ownership
{
    std::unique_ptr<MyClass> ptr1 = std::make_unique<MyClass>(10);
    // Automatically deleted when ptr1 goes out of scope
    // Cannot be copied, only moved

    auto ptr2 = std::make_unique<MyClass>(20);  // Using auto
    std::unique_ptr<MyClass> ptr3 = std::move(ptr2);  // Transfer ownership
    // ptr2 is now nullptr
}

// std::shared_ptr - shared ownership
{
    std::shared_ptr<MyClass> ptr1 = std::make_shared<MyClass>(30);
    {
        std::shared_ptr<MyClass> ptr2 = ptr1;  // Both own the object
        std::cout << "Reference count: " << ptr1.use_count() << std::endl;  // 2
    }  // ptr2 destroyed, object still exists
    std::cout << "Reference count: " << ptr1.use_count() << std::endl;  // 1
}  // Object deleted when last shared_ptr is destroyed

// Array with smart pointers (C++17)
auto arr = std::make_unique<MyClass[]>(5);
```

**Advantages:**
- Automatic memory management
- Exception-safe
- Clear ownership semantics

**Disadvantages:**
- Slight overhead (especially shared_ptr)
- Reference counting overhead

### 4. **Initialization Patterns**

C++ offers various initialization syntaxes with different behaviors.

```cpp
class Point {
public:
    int x, y;
    Point() : x(0), y(0) {}
    Point(int x, int y) : x(x), y(y) {}
};

// Default initialization
Point p1;  // Calls default constructor: Point()

// Direct initialization
Point p2(10, 20);  // Calls Point(int, int)

// Copy initialization
Point p3 = Point(30, 40);  // May involve copy/move

// List initialization (Uniform initialization - C++11)
Point p4{50, 60};        // Direct list initialization
Point p5 = {70, 80};     // Copy list initialization
auto p6 = Point{90, 100}; // With auto

// Value initialization
Point p7{};     // Zero-initializes: x=0, y=0
Point* p8 = new Point();   // Value initialization on heap

// Aggregate initialization (for POD types)
struct Data {
    int a;
    double b;
    char c;
};

Data d1 = {1, 2.5, 'x'};   // C-style
Data d2{1, 2.5, 'x'};      // C++11 style
Data d3{.a=1, .b=2.5};     // C++20 designated initializers
```

### 5. **Constructor Patterns**

Different ways to call constructors for initialization.

```cpp
class Resource {
private:
    int* data;
    size_t size;

public:
    // Default constructor
    Resource() : data(nullptr), size(0) {
        std::cout << "Default constructor" << std::endl;
    }

    // Parameterized constructor
    Resource(size_t sz) : data(new int[sz]), size(sz) {
        std::cout << "Parameterized constructor" << std::endl;
    }

    // Copy constructor
    Resource(const Resource& other) : size(other.size) {
        data = new int[size];
        std::copy(other.data, other.data + size, data);
        std::cout << "Copy constructor" << std::endl;
    }

    // Move constructor (C++11)
    Resource(Resource&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
        std::cout << "Move constructor" << std::endl;
    }

    // Destructor
    ~Resource() {
        delete[] data;
        std::cout << "Destructor" << std::endl;
    }
};

// Usage examples
Resource r1;                      // Default constructor
Resource r2(100);                 // Parameterized constructor
Resource r3 = r2;                 // Copy constructor
Resource r4 = std::move(r2);      // Move constructor
Resource r5(std::move(r3));       // Move constructor (explicit)
```

### 6. **Factory Pattern**

Using factory functions for object creation.

```cpp
class Shape {
public:
    virtual void draw() = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
    double radius;
public:
    Circle(double r) : radius(r) {}
    void draw() override { std::cout << "Drawing circle" << std::endl; }
};

class Rectangle : public Shape {
    double width, height;
public:
    Rectangle(double w, double h) : width(w), height(h) {}
    void draw() override { std::cout << "Drawing rectangle" << std::endl; }
};

// Factory function
std::unique_ptr<Shape> createShape(const std::string& type) {
    if (type == "circle") {
        return std::make_unique<Circle>(5.0);
    } else if (type == "rectangle") {
        return std::make_unique<Rectangle>(4.0, 6.0);
    }
    return nullptr;
}

// Usage
auto shape = createShape("circle");
if (shape) {
    shape->draw();
}
```

### 7. **Placement New**

Constructing objects at a specific memory location.

```cpp
#include <new>

// Pre-allocated buffer
alignas(MyClass) char buffer[sizeof(MyClass)];

// Construct object in buffer
MyClass* obj = new (buffer) MyClass(42);

// Use object
obj->value = 100;

// Must manually call destructor
obj->~MyClass();

// Common use case: memory pools
class MemoryPool {
    char buffer[1024];
public:
    template<typename T, typename... Args>
    T* construct(Args&&... args) {
        void* ptr = /* allocate from buffer */;
        return new (ptr) T(std::forward<Args>(args)...);
    }
};
```

### 8. **Array Initialization Patterns**

Different ways to create and initialize arrays of objects.

```cpp
// Stack arrays
MyClass arr1[3];                    // Default constructor for each
MyClass arr2[3] = {MyClass(1), MyClass(2), MyClass(3)};  // Specific initialization
MyClass arr3[] = {MyClass(10), MyClass(20)};  // Size inferred

// Uniform initialization (C++11)
MyClass arr4[3] = {{1}, {2}, {3}};
MyClass arr5[3]{{1}, {2}, {3}};

// Heap arrays
MyClass* heap_arr1 = new MyClass[5];      // Default constructor
delete[] heap_arr1;

// std::array (C++11)
#include <array>
std::array<MyClass, 3> arr6 = {MyClass(1), MyClass(2), MyClass(3)};
std::array<MyClass, 3> arr7{MyClass(1), MyClass(2), MyClass(3)};

// std::vector (dynamic array)
#include <vector>
std::vector<MyClass> vec1;                // Empty vector
std::vector<MyClass> vec2(5);             // 5 default-constructed objects
std::vector<MyClass> vec3(5, MyClass(42)); // 5 copies of MyClass(42)
std::vector<MyClass> vec4{MyClass(1), MyClass(2), MyClass(3)};  // Initializer list
```

### 9. **Emplace Construction**

Constructing objects in-place within containers (C++11).

```cpp
#include <vector>
#include <map>

std::vector<MyClass> vec;

// push_back creates temporary and moves/copies it
vec.push_back(MyClass(10));

// emplace_back constructs directly in the vector (more efficient)
vec.emplace_back(20);  // Constructs MyClass(20) in-place

// Similarly for maps
std::map<int, MyClass> myMap;
myMap.emplace(1, MyClass(100));        // Creates pair in-place
myMap.try_emplace(2, 200);             // Even better, doesn't construct if key exists

// emplace with multiple arguments
struct Person {
    std::string name;
    int age;
    Person(std::string n, int a) : name(n), age(a) {}
};

std::vector<Person> people;
people.emplace_back("Alice", 30);  // Constructs Person directly in vector
```

### 10. **RAII Pattern (Resource Acquisition Is Initialization)**

Tying resource lifetime to object lifetime.

```cpp
class FileHandler {
    FILE* file;
public:
    // Resource acquired in constructor
    FileHandler(const char* filename, const char* mode) {
        file = fopen(filename, mode);
        if (!file) throw std::runtime_error("Failed to open file");
    }

    // Resource released in destructor
    ~FileHandler() {
        if (file) {
            fclose(file);
        }
    }

    // Prevent copying
    FileHandler(const FileHandler&) = delete;
    FileHandler& operator=(const FileHandler&) = delete;

    // Allow moving
    FileHandler(FileHandler&& other) noexcept : file(other.file) {
        other.file = nullptr;
    }

    FILE* get() { return file; }
};

// Usage - no need to manually close file
void processFile() {
    FileHandler handler("data.txt", "r");
    // Use handler.get()...
    // File automatically closed when handler goes out of scope
}
```

### 11. **Copy Elision and RVO (Return Value Optimization)**

The compiler can optimize away unnecessary copies.

```cpp
MyClass createObject() {
    MyClass obj(100);
    return obj;  // RVO: object constructed directly in caller's space
}

MyClass obj1 = createObject();  // No copy/move, direct construction (C++17 guaranteed)

// Named Return Value Optimization (NRVO)
MyClass createNamed(int value) {
    MyClass result(value);
    // ... operations on result
    return result;  // May be optimized (not guaranteed)
}
```

### Best Practices for Object Instantiation

1. **Prefer stack allocation** when possible - it's fastest and safest
2. **Use smart pointers** instead of raw new/delete for heap allocation
3. **Use `std::make_unique` and `std::make_shared`** for creating smart pointers
4. **Use uniform initialization `{}`** to avoid most vexing parse and narrowing conversions
5. **Use `emplace` methods** in containers for in-place construction
6. **Follow RAII principles** for resource management
7. **Prefer `std::vector` and `std::array`** over raw arrays
8. **Avoid naked `new`** - use smart pointers or containers

```cpp
// Good practices example
void goodPractices() {
    // Stack allocation when lifetime is scoped
    MyClass local(42);

    // Smart pointers for heap allocation
    auto ptr = std::make_unique<MyClass>(100);

    // Uniform initialization
    MyClass obj{50};

    // Containers for collections
    std::vector<MyClass> vec;
    vec.emplace_back(10);
    vec.emplace_back(20);

    // RAII for resources
    std::ifstream file("data.txt");
    // File automatically closed
}
```

## C++ Strings and Their Methods

In C++, the `std::string` class provides a powerful and flexible way to handle strings. It offers a variety of methods for string manipulation, making it easier to perform common operations without dealing with low-level character arrays. Below are some of the most commonly used `std::string` methods in detail:

### 1. **Constructors**

`std::string` offers multiple constructors to initialize strings in different ways.

```cpp
#include <string>

// Default constructor
std::string str1;

// Constructor with a C-string
std::string str2("Hello, World!");

// Constructor with a specific number of repeated characters
std::string str3(5, 'a'); // "aaaaa"

// Copy constructor
std::string str4(str2);

// Substring constructor
std::string str5(str2, 7, 5); // "World"
```

### 2. **Size and Capacity**

- `size()` / `length()`: Returns the number of characters in the string.
- `capacity()`: Returns the size of the storage space currently allocated for the string.

```cpp
std::string str = "Example";
size_t len = str.size(); // 7
size_t cap = str.capacity(); // Implementation-defined
```

### 3. **Accessing Characters**

- `operator[]`: Accesses character at a specific index.
- `at()`: Accesses character at a specific index with bounds checking.
- `front()` / `back()`: Accesses the first and last characters.

```cpp
std::string str = "Hello";
char ch = str[1]; // 'e'
char ch_at = str.at(2); // 'l'
char first = str.front(); // 'H'
char last = str.back(); // 'o'
```

### 4. **Modifiers**

- `append()`: Adds characters to the end of the string.
- `clear()`: Removes all characters from the string.
- `insert()`: Inserts characters at a specified position.
- `erase()`: Removes characters from a specified position.
- `replace()`: Replaces part of the string with another string.

```cpp
std::string str = "Hello";
str.append(", World!"); // "Hello, World!"
str.insert(5, " C++"); // "Hello C++, World!"
str.erase(5, 6); // "HelloWorld!"
str.replace(5, 5, " C++"); // "Hello C++!"
str.clear(); // ""
```

### 5. **Substring and Extracting**

- `substr()`: Returns a substring starting from a specified position.

```cpp
std::string str = "Hello, World!";
std::string sub = str.substr(7, 5); // "World"
```

### 6. **Finding Characters and Substrings**

- `find()`: Searches for a substring or character and returns the position.
- `rfind()`: Searches for a substring or character from the end.

```cpp
std::string str = "Hello, World!";
size_t pos = str.find("World"); // 7
size_t rpos = str.rfind('o'); // 8
```

### 7. **Comparison**

- `compare()`: Compares two strings.

```cpp
std::string str1 = "apple";
std::string str2 = "banana";

int result = str1.compare(str2);
// result < 0 since "apple" < "banana"
```

### 8. **Conversion to C-string**

- `c_str()`: Returns a C-style null-terminated string.

```cpp
std::string str = "Hello";
const char* cstr = str.c_str();
```

### 9. **Iterators**

`std::string` supports iterators to traverse the string.

```cpp
std::string str = "Hello";
for (std::string::iterator it = str.begin(); it != str.end(); ++it) {
    std::cout << *it << ' ';
}
// Output: H e l l o 
```

### 10. **Emplace and Emplace_back**

- `emplace()`: Constructs and inserts a substring.
- `emplace_back()`: Appends a character to the end of the string.

```cpp
std::string str = "Hello";
str.emplace(str.size(), '!'); // "Hello!"
str.emplace_back('?'); // "Hello!?"
```

### 11. **Swap**

- `swap()`: Swaps the contents of two strings.

```cpp
std::string str1 = "Hello";
std::string str2 = "World";
str1.swap(str2);
// str1 is now "World", str2 is now "Hello"
```

### 12. **Transform**

You can apply transformations to each character using algorithms.

```cpp
#include <algorithm>

std::string str = "Hello";
std::transform(str.begin(), str.end(), str.begin(), ::toupper); // "HELLO"
```

### 13. **Other Useful Methods**

- `empty()`: Checks if the string is empty.
- `find_first_of()` / `find_last_of()`: Finds the first/last occurrence of any character from a set.
- `find_first_not_of()` / `find_last_not_of()`: Finds the first/last character not in a set.

```cpp
std::string str = "Hello";
bool isEmpty = str.empty(); // false
size_t pos = str.find_first_of('e'); // 1
size_t not_pos = str.find_first_not_of('H'); // 1
```

### Example Usage

```cpp
#include <iostream>
#include <string>

int main() {
    std::string greeting = "Hello";
    greeting += ", World!"; // Using operator +=
    
    std::cout << greeting << std::endl; // Output: Hello, World!
    
    // Find and replace
    size_t pos = greeting.find("World");
    if (pos != std::string::npos) {
        greeting.replace(pos, 5, "C++");
    }
    
    std::cout << greeting << std::endl; // Output: Hello, C++!
    
    return 0;
}
```

Understanding and utilizing these `std::string` methods can greatly enhance your ability to manipulate and manage text in C++ applications effectively.




## C++ Vectors and Their Methods

In C++, the `std::vector` class template provides a dynamic array that can resize itself automatically when elements are added or removed. It offers numerous methods to manipulate the data efficiently. Below are detailed explanations and examples of various `std::vector` methods:

### 1. **Constructors**

`std::vector` offers multiple constructors to initialize vectors in different ways.

```cpp
#include <vector>

// Default constructor
std::vector<int> vec1;

// Constructor with a specific size
std::vector<int> vec2(5); // {0, 0, 0, 0, 0}

// Constructor with a specific size and initial value
std::vector<int> vec3(5, 10); // {10, 10, 10, 10, 10}

// Initializer list constructor
std::vector<int> vec4 = {1, 2, 3, 4, 5};

// Copy constructor
std::vector<int> vec5(vec4);
```

### 2. **Size and Capacity**

- `size()`: Returns the number of elements in the vector.
- `capacity()`: Returns the size of the storage space currently allocated for the vector, expressed in terms of elements.
- `empty()`: Checks whether the vector is empty.

```cpp
std::vector<int> vec = {1, 2, 3};
size_t sz = vec.size(); // 3
size_t cap = vec.capacity(); // >= 3
bool isEmpty = vec.empty(); // false
```

### 3. **Element Access**

- `operator[]`: Accesses element at a specific index without bounds checking.
- `at()`: Accesses element at a specific index with bounds checking.
- `front()`: Accesses the first element.
- `back()`: Accesses the last element.
- `data()`: Returns a pointer to the underlying array.

```cpp
std::vector<int> vec = {10, 20, 30, 40, 50};
int first = vec[0]; // 10
int third = vec.at(2); // 30
int front = vec.front(); // 10
int back = vec.back(); // 50
int* ptr = vec.data(); // Pointer to the first element
```

### 4. **Modifiers**

- `push_back()`: Adds an element to the end of the vector.
- `pop_back()`: Removes the last element of the vector.
- `insert()`: Inserts elements at a specified position.
- `erase()`: Removes elements from a specified position or range.
- `clear()`: Removes all elements from the vector.
- `resize()`: Changes the number of elements stored.
- `shrink_to_fit()`: Reduces capacity to fit the size.

```cpp
std::vector<int> vec = {1, 2, 3};

// push_back
vec.push_back(4); // {1, 2, 3, 4}

// pop_back
vec.pop_back(); // {1, 2, 3}

// insert
vec.insert(vec.begin() + 1, 10); // {1, 10, 2, 3}

// erase single element
vec.erase(vec.begin() + 2); // {1, 10, 3}

// erase range
vec.erase(vec.begin(), vec.begin() + 1); // {10, 3}

// clear
vec.clear(); // {}

// resize
vec.resize(5, 100); // {100, 100, 100, 100, 100}

// shrink_to_fit
vec.shrink_to_fit();
```

### 5. **Iterators**

Vectors support iterators to traverse and manipulate elements.

- `begin()`: Returns an iterator to the first element.
- `end()`: Returns an iterator to one past the last element.
- `rbegin()`: Returns a reverse iterator to the last element.
- `rend()`: Returns a reverse iterator to one before the first element.

```cpp
std::vector<int> vec = {1, 2, 3, 4, 5};

// Forward iteration
for(auto it = vec.begin(); it != vec.end(); ++it) {
    std::cout << *it << " ";
}

// Reverse iteration
for(auto it = vec.rbegin(); it != vec.rend(); ++it) {
    std::cout << *it << " ";
}
```

### 6. **Algorithms Support**

Vectors work seamlessly with standard algorithms from the C++ Standard Library.

```cpp
#include <algorithm>

std::vector<int> vec = {5, 3, 1, 4, 2};

// Sort the vector
std::sort(vec.begin(), vec.end()); // {1, 2, 3, 4, 5}

// Reverse the vector
std::reverse(vec.begin(), vec.end()); // {5, 4, 3, 2, 1}

// Find an element
auto it = std::find(vec.begin(), vec.end(), 3);
if(it != vec.end()) {
    std::cout << "Found: " << *it << std::endl;
}
```

### 7. **Capacity Management**

- `reserve()`: Increases the capacity of the vector to a value that's greater or equal to the specified.
- `capacity()`: Explained earlier.

```cpp
std::vector<int> vec;
vec.reserve(100); // Reserve space for 100 elements
std::cout << "Capacity: " << vec.capacity() << std::endl;
```

Understanding and utilizing `std::vector` and its various methods can significantly enhance the efficiency and flexibility of your C++ programs, allowing for dynamic memory management and rich data manipulation capabilities.




### 4. **Maps**

C++ provides the `std::map` container, which is an associative container that stores elements formed by a combination of a key and a value. `std::map` automatically sorts its elements by key and allows fast retrieval of individual elements based on their keys.

#### Constructors

`std::map` offers multiple constructors to initialize maps in different ways.

```cpp
#include <map>
#include <string>

// Default constructor
std::map<int, std::string> map1;

// Initializer list constructor
std::map<int, std::string> map2 = {
    {1, "one"},
    {2, "two"},
    {3, "three"}
};

// Range constructor
std::vector<std::pair<int, std::string>> vec = { {4, "four"}, {5, "five"} };
std::map<int, std::string> map3(vec.begin(), vec.end());

// Copy constructor
std::map<int, std::string> map4(map2);
```

#### Size and Capacity

- `size()`: Returns the number of elements in the map.
- `empty()`: Checks whether the map is empty.

```cpp
std::map<int, std::string> map = { {1, "one"}, {2, "two"}, {3, "three"} };
size_t sz = map.size(); // 3
bool isEmpty = map.empty(); // false
```

#### Element Access

- `operator[]`: Accesses or inserts elements with the given key.
- `at()`: Accesses elements with bounds checking.
- `find()`: Finds an element with a specific key.
- `count()`: Returns the number of elements with a specific key.

```cpp
// Using operator[]
map[4] = "four"; // Inserts if key 4 does not exist

// Using at()
try {
    std::string value = map.at(2); // "two"
} catch(const std::out_of_range& e) {
    // Handle error
}

// Using find()
auto it = map.find(3);
if(it != map.end()) {
    std::cout << "Found: " << it->second << std::endl; // "three"
}

// Using count()
if(map.count(5)) {
    std::cout << "Key 5 exists." << std::endl;
} else {
    std::cout << "Key 5 does not exist." << std::endl;
}
```

#### Inserting Elements

- `insert()`: Inserts elements into the map.
- `emplace()`: Constructs elements in-place.

```cpp
// Using insert()
map.insert({1, "one"});
map.insert(std::pair<int, std::string>(2, "two"));

// Using emplace()
map.emplace(3, "three");
```

#### Deleting Elements

- `erase()`: Removes elements by key or iterator.
- `clear()`: Removes all elements from the map.

```cpp
std::map<int, std::string> map = { {1, "one"}, {2, "two"}, {3, "three"} };

// Erase by key
map.erase(2);

// Erase by iterator
auto itErase = map.find(3);
if(itErase != map.end()) {
    map.erase(itErase);
}

// Clear all elements
map.clear();
```

#### Iterating Through a Map

```cpp
std::map<int, std::string> map = { {1, "one"}, {2, "two"}, {3, "three"} };

// Using iterator
for(auto it = map.begin(); it != map.end(); ++it) {
    std::cout << it->first << ": " << it->second << std::endl;
}

// Using range-based for loop
for(const auto& pair : map) {
    std::cout << pair.first << ": " << pair.second << std::endl;
}
```

Understanding and utilizing `std::map` and its various methods can greatly enhance your ability to manage key-value pairs efficiently in C++ applications.


### 4. **Smart Pointers**

Smart pointers in C++ are template classes provided by the Standard Library that facilitate automatic and exception-safe memory management. They help manage dynamically allocated objects by ensuring that resources are properly released when they are no longer needed, thus preventing memory leaks and other related issues. C++ offers several types of smart pointers, each tailored to specific use cases and ownership semantics.

#### Types of Smart Pointers

1. **`std::unique_ptr`**
2. **`std::shared_ptr`**
3. **`std::weak_ptr`**

---

#### 1. `std::unique_ptr`

`std::unique_ptr` is a smart pointer that owns and manages another object through a pointer and disposes of that object when the `unique_ptr` goes out of scope. It ensures exclusive ownership, meaning that there can be only one `unique_ptr` instance owning a particular object at any given time.

**Key Characteristics:**
- **Exclusive Ownership:** Only one `std::unique_ptr` can own the object at a time.
- **No Copying:** `unique_ptr` cannot be copied to prevent multiple ownerships. However, it can be moved.
- **Lightweight:** Minimal overhead compared to raw pointers.

**Usage Example:**

```cpp
#include <memory>
#include <iostream>

int main() {
    // Creating a unique_ptr to an integer
    std::unique_ptr<int> ptr1(new int(10));
    std::cout << "Value: " << *ptr1 << std::endl; // Output: Value: 10

    // Transferring ownership using std::move
    std::unique_ptr<int> ptr2 = std::move(ptr1);
    if (!ptr1) {
        std::cout << "ptr1 is now null." << std::endl;
    }
    std::cout << "Value: " << *ptr2 << std::endl; // Output: Value: 10

    // Automatic deletion when ptr2 goes out of scope
    return 0;
}
```

**Common Methods:**
- **`get()`**: Returns the raw pointer.
- **`release()`**: Releases ownership of the managed object and returns the pointer.
- **`reset()`**: Deletes the currently managed object and takes ownership of a new one.
- **`operator*` and `operator->`**: Dereference operators to access the managed object.

---

#### 2. `std::shared_ptr`

`std::shared_ptr` is a smart pointer that maintains shared ownership of an object through a pointer. Multiple `shared_ptr` instances can own the same object, and the object is destroyed only when the last `shared_ptr` owning it is destroyed or reset.

**Key Characteristics:**
- **Shared Ownership:** Multiple `shared_ptr` instances can own the same object.
- **Reference Counting:** Keeps track of how many `shared_ptr` instances own the object.
- **Thread-Safe Reference Counting:** Safe to use in multi-threaded applications for reference counting operations.

**Usage Example:**

```cpp
#include <memory>
#include <iostream>

int main() {
    // Creating a shared_ptr to an integer
    std::shared_ptr<int> ptr1 = std::make_shared<int>(20);
    std::cout << "Value: " << *ptr1 << ", Count: " << ptr1.use_count() << std::endl; // Output: Value: 20, Count: 1

    // Creating another shared_ptr sharing the same object
    std::shared_ptr<int> ptr2 = ptr1;
    std::cout << "Value: " << *ptr2 << ", Count: " << ptr1.use_count() << std::endl; // Output: Value: 20, Count: 2

    // Resetting ptr1
    ptr1.reset();
    std::cout << "ptr1 reset. Count: " << ptr2.use_count() << std::endl; // Output: Count: 1

    // Automatic deletion when ptr2 goes out of scope
    return 0;
}
```

**Common Methods:**
- **`use_count()`**: Returns the number of `shared_ptr` instances sharing ownership.
- **`unique()`**: Checks if the `shared_ptr` is the only owner.
- **`reset()`**: Releases ownership of the managed object.
- **`swap()`**: Exchanges the managed object with another `shared_ptr`.

---

#### 3. `std::weak_ptr`

`std::weak_ptr` is a smart pointer that holds a non-owning ("weak") reference to an object that is managed by `std::shared_ptr`. It is used to prevent circular references that can lead to memory leaks by allowing one part of the code to observe an object without affecting its lifetime.

**Key Characteristics:**
- **Non-Owning:** Does not contribute to the reference count.
- **Avoids Circular References:** Useful in scenarios like bidirectional relationships.
- **Access Controlled:** Must be converted to `std::shared_ptr` to access the managed object.

**Usage Example:**

```cpp
#include <memory>
#include <iostream>

struct Node {
    int value;
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev; // Using weak_ptr to prevent circular reference

    Node(int val) : value(val), next(nullptr), prev() {}
};

int main() {
    auto node1 = std::make_shared<Node>(1);
    auto node2 = std::make_shared<Node>(2);

    node1->next = node2;
    node2->prev = node1; // weak_ptr does not increase reference count

    std::cout << "Node1 value: " << node1->value << std::endl;
    std::cout << "Node2 value: " << node2->value << std::endl;

    // Accessing the previous node
    if(auto prev = node2->prev.lock()) {
        std::cout << "Node2's previous node value: " << prev->value << std::endl;
    } else {
        std::cout << "Previous node no longer exists." << std::endl;
    }

    return 0;
}
```

**Common Methods:**
- **`lock()`**: Attempts to acquire a `std::shared_ptr` to the managed object.
- **`expired()`**: Checks if the managed object has been deleted.
- **`reset()`**: Releases the managed object reference.

---

#### Common Methods Across Smart Pointers

While each smart pointer type has its specific methods, there are several common methods that they share:

- **`get()`**: Returns the raw pointer managed by the smart pointer.
  
  ```cpp
  std::unique_ptr<int> ptr = std::make_unique<int>(100);
  int* rawPtr = ptr.get();
  std::cout << "Raw pointer value: " << *rawPtr << std::endl; // Output: 100
  ```

- **`reset()`**: Releases the ownership of the managed object and optionally takes ownership of a new object.
  
  ```cpp
  std::shared_ptr<int> ptr = std::make_shared<int>(200);
  ptr.reset(new int(300)); // Old object is deleted, ptr now owns the new object
  std::cout << "New value: " << *ptr << std::endl; // Output: 300
  ```

- **`swap()`**: Exchanges the managed objects of two smart pointers.
  
  ```cpp
  std::unique_ptr<int> ptr1 = std::make_unique<int>(400);
  std::unique_ptr<int> ptr2 = std::make_unique<int>(500);
  ptr1.swap(ptr2);
  std::cout << "ptr1: " << *ptr1 << ", ptr2: " << *ptr2 << std::endl; // Output: ptr1: 500, ptr2: 400
  ```

- **Dereference Operators (`*` and `->`)**: Access the managed object.
  
  ```cpp
  std::shared_ptr<std::string> ptr = std::make_shared<std::string>("Hello");
  std::cout << "String: " << *ptr << std::endl; // Output: Hello
  std::cout << "String length: " << ptr->length() << std::endl; // Output: 5
  ```

---

#### Best Practices

- **Prefer `std::make_unique` and `std::make_shared`:** These functions are exception-safe and more efficient.
  
  ```cpp
  auto ptr = std::make_unique<MyClass>();
  auto sharedPtr = std::make_shared<MyClass>();
  ```

- **Use `std::unique_ptr` When Ownership is Exclusive:** It clearly signifies ownership semantics and incurs no overhead of reference counting.
  
  ```cpp
  std::unique_ptr<Resource> resource = std::make_unique<Resource>();
  ```

- **Use `std::shared_ptr` When Ownership is Shared:** Useful in scenarios where multiple parts of the program need to share access to the same resource.
  
  ```cpp
  std::shared_ptr<Logger> logger1 = std::make_shared<Logger>();
  std::shared_ptr<Logger> logger2 = logger1;
  ```

- **Avoid `std::shared_ptr` Unless Necessary:** It introduces overhead due to reference counting. Use it only when shared ownership is required.
  
- **Break Circular References with `std::weak_ptr`:** When two objects share ownership via `std::shared_ptr`, use `std::weak_ptr` to prevent memory leaks.
  
  ```cpp
  struct A {
      std::shared_ptr<B> b_ptr;
  };
  
  struct B {
      std::weak_ptr<A> a_ptr; // weak_ptr breaks the circular reference
  };
  ```

---

Understanding and effectively utilizing smart pointers is crucial for modern C++ programming. They not only simplify memory management but also enhance the safety and performance of applications by preventing common issues related to dynamic memory allocation.

### 5. **`std::function` and `std::bind`**

`std::function` and `std::bind` are powerful utilities in the C++ Standard Library that facilitate higher-order programming by allowing functions to be treated as first-class objects. They enable the storage, modification, and invocation of functions in a flexible and generic manner, enhancing the capabilities of callback mechanisms, event handling, and functional programming paradigms in C++.

#### `std::function`

`std::function` is a versatile, type-erased function wrapper that can store any callable target—such as free functions, member functions, lambda expressions, or other function objects—provided they match a specific function signature. This flexibility makes it an essential tool for designing callback interfaces and managing dynamic function invocation.

**Key Characteristics:**
- **Type-Erasure:** Abstracts away the specific type of the callable, allowing different types of callable objects to be stored in the same `std::function` variable.
- **Copyable and Assignable:** `std::function` instances can be copied and assigned, enabling their use in standard containers and algorithms.
- **Invoke Any Callable:** Can represent free functions, member functions, lambda expressions, and function objects.

**Basic Usage Example:**

```cpp
#include <functional>
#include <iostream>

// A free function
int add(int a, int b) {
    return a + b;
}

int main() {
    // Storing a free function in std::function
    std::function<int(int, int)> func = add;
    std::cout << "add(2, 3) = " << func(2, 3) << std::endl; // Output: 5

    // Storing a lambda expression
    std::function<int(int, int)> lambdaFunc = [](int a, int b) -> int {
        return a * b;
    };
    std::cout << "lambdaFunc(2, 3) = " << lambdaFunc(2, 3) << std::endl; // Output: 6

    // Storing a member function (requires binding)
    struct Calculator {
        int subtract(int a, int b) const {
            return a - b;
        }
    };

    Calculator calc;
    std::function<int(int, int)> memberFunc = std::bind(&Calculator::subtract, &calc, std::placeholders::_1, std::placeholders::_2);
    std::cout << "calc.subtract(5, 3) = " << memberFunc(5, 3) << std::endl; // Output: 2

    return 0;
}
```

**Common Methods:**
- **`operator()`**: Invokes the stored callable.
- **`target()`**: Retrieves a pointer to the stored callable if it matches a specific type.
- **`reset()`**: Clears the stored callable, making the `std::function` empty.

#### `std::bind`

`std::bind` is a utility that allows you to create a new function object by binding some or all of the arguments of an existing function to specific values. This is particularly useful for adapting functions to match desired interfaces or for creating callbacks with pre-specified arguments.

**Key Characteristics:**
- **Argument Binding:** Fixes certain arguments of a function, producing a new function object with fewer parameters.
- **Placeholders:** Uses placeholders like `std::placeholders::_1` to indicate arguments that will be provided later.
- **Supports Various Callables:** Can bind free functions, member functions, and function objects.

**Basic Usage Example:**

```cpp
#include <functional>
#include <iostream>

// A free function
int multiply(int a, int b) {
    return a * b;
}

struct Calculator {
    int divide(int a, int b) const {
        if(b == 0) throw std::invalid_argument("Division by zero");
        return a / b;
    }
};

int main() {
    // Binding the first argument of multiply to 5
    auto timesFive = std::bind(multiply, 5, std::placeholders::_1);
    std::cout << "multiply(5, 4) = " << timesFive(4) << std::endl; // Output: 20

    // Binding a member function with the object instance
    Calculator calc;
    auto divideBy = std::bind(&Calculator::divide, &calc, std::placeholders::_1, 2);
    std::cout << "calc.divide(10, 2) = " << divideBy(10) << std::endl; // Output: 5

    return 0;
}
```

**Common Use Cases:**
- **Creating Callbacks:** Adapting functions to match callback interfaces that require a specific signature.
- **Event Handling:** Binding member functions of objects to event handlers with predefined arguments.
- **Functional Programming:** Enabling partial application and currying of functions for more functional-style code.

**Advanced Usage Example:**

```cpp
#include <functional>
#include <iostream>
#include <vector>

class Logger {
public:
    void log(const std::string& message, int level) const {
        std::cout << "Level " << level << ": " << message << std::endl;
    }
};

int main() {
    Logger logger;

    // Binding the logger object and log level to create a simplified log function
    auto infoLog = std::bind(&Logger::log, &logger, std::placeholders::_1, 1);
    auto errorLog = std::bind(&Logger::log, &logger, std::placeholders::_1, 3);

    infoLog("This is an informational message."); // Output: Level 1: This is an informational message.
    errorLog("This is an error message."); // Output: Level 3: This is an error message.

    // Storing bind expressions in a std::vector of std::function
    std::vector<std::function<void(const std::string&)>> logs;
    logs.push_back(infoLog);
    logs.push_back(errorLog);

    for(auto& logFunc : logs) {
        logFunc("Logging through stored function.");
    }
    // Output:
    // Level 1: Logging through stored function.
    // Level 3: Logging through stored function.

    return 0;
}
```

**Best Practices:**
- **Prefer Lambda Expressions Over `std::bind`:** Lambdas often provide clearer and more readable syntax compared to `std::bind`.
  
  ```cpp
  // Using std::bind
  auto timesFive = std::bind(multiply, 5, std::placeholders::_1);
  
  // Equivalent using a lambda
  auto timesFiveLambda = [](int a) -> int {
      return multiply(5, a);
  };
  ```
  
- **Use `std::function` for Flexibility:** When storing or passing callable objects that may vary in type, use `std::function` to accommodate different callables.

- **Avoid Unnecessary Bindings:** Excessive use of `std::bind` can lead to less readable code. Assess whether a lambda or a direct function call may be more appropriate.

By leveraging `std::function` and `std::bind`, developers can create more abstract, flexible, and reusable code components, facilitating sophisticated callback mechanisms and enhancing the expressive power of C++.



## C++ in Competitive Programming

Competitive programming demands not only a deep understanding of algorithms and data structures but also the ability to implement them efficiently within strict time and memory constraints. C++ is a favored language in this arena due to its performance, rich Standard Template Library (STL), and powerful language features. Below are various methods and techniques in C++ that are extensively used in competitive programming:

### 1. **Fast Input/Output**

Efficient handling of input and output can significantly reduce execution time, especially with large datasets.

- **Untie C++ Streams from C Streams:**
  ```cpp
  std::ios::sync_with_stdio(false);
  std::cin.tie(nullptr);
  ```
  Disabling the synchronization between C and C++ standard streams and untieing `cin` from `cout` can speed up I/O operations.

- **Use of `scanf` and `printf`:**
  For even faster I/O, some competitors prefer using C-style I/O functions.

### 2. **Utilizing the Standard Template Library (STL)**

The STL provides a suite of ready-to-use data structures and algorithms that can save time and reduce the likelihood of bugs.

- **Vectors (`std::vector`):**
  Dynamic arrays that allow for efficient random access and dynamic resizing.
  ```cpp
  std::vector<int> vec = {1, 2, 3};
  vec.push_back(4);
  ```

- **Pairs and Tuples (`std::pair`, `std::tuple`):**
  Useful for storing multiple related values.
  ```cpp
  std::pair<int, int> p = {1, 2};
  std::tuple<int, int, int> t = {1, 2, 3};
  ```

- **Sets and Maps (`std::set`, `std::map`):**
  Efficiently handle unique elements and key-value associations.

- **Algorithms (`std::sort`, `std::binary_search`, etc.):**
  Implement common algorithms with optimized performance.

### 3. **Graph Representations and Algorithms**

Graphs are a staple in competitive programming problems. Efficient representation and traversal are crucial.

- **Adjacency List:**
  ```cpp
  int n; // Number of nodes
  std::vector<std::vector<int>> adj(n + 1);
  adj[u].push_back(v);
  adj[v].push_back(u); // For undirected graphs
  ```

- **Depth-First Search (DFS) and Breadth-First Search (BFS):**
  Fundamental traversal techniques.

- **Dijkstra's and Floyd-Warshall Algorithms:**
  For shortest path problems.

### 4. **Dynamic Programming (DP)**

DP is essential for solving optimization problems by breaking them down into simpler subproblems.

- **Memoization and Tabulation:**
  ```cpp
  // Example of Fibonacci using memoization
  long long fib(int n, std::vector<long long> &dp) {
      if(n <= 1) return n;
      if(dp[n] != -1) return dp[n];
      return dp[n] = fib(n-1, dp) + fib(n-2, dp);
  }
  ```

- **State Optimization:**
  Reducing space complexity by optimizing states.

### 5. **Greedy Algorithms**

These algorithms make the locally optimal choice at each step with the hope of finding the global optimum.

- **Interval Scheduling:**
  Selecting the maximum number of non-overlapping intervals.

- **Huffman Coding:**
  For efficient encoding.

### 6. **Bit Manipulation**

Bitwise operations can optimize certain calculations and are useful in problems involving subsets or binary representations.

- **Common Operations:**
  - Setting a bit: `x | (1 << pos)`
  - Clearing a bit: `x & ~(1 << pos)`
  - Toggling a bit: `x ^ (1 << pos)`

- **Bitmask DP:**
  Using bitmasks to represent states in DP.

### 7. **Number Theory**

Many problems involve mathematical concepts such as primes, GCD, and modular arithmetic.

- **Sieve of Eratosthenes:**
  For finding all prime numbers up to a certain limit.
  ```cpp
  std::vector<bool> is_prime(n+1, true);
  is_prime[0] = is_prime[1] = false;
  for(int i=2; i*i <= n; ++i){
      if(is_prime[i]){
          for(int j=i*i; j<=n; j+=i){
              is_prime[j] = false;
          }
      }
  }
  ```

- **Modular Exponentiation:**
  Efficiently computing large exponents under a modulus.
  ```cpp
  long long power(long long a, long long b, long long mod){
      long long res = 1;
      a %= mod;
      while(b > 0){
          if(b & 1) res = res * a % mod;
          a = a * a % mod;
          b >>= 1;
      }
      return res;
  }
  ```

### 8. **String Algorithms**

Handling and processing strings efficiently is vital in many problems.

- **KMP Algorithm:**
  For pattern matching with linear time complexity.

- **Trie Data Structure:**
  Efficiently storing and searching a dynamic set of strings.

### 9. **Data Structures**

Choosing the right data structure can make or break your solution.

- **Segment Trees and Binary Indexed Trees (Fenwick Trees):**
  For range queries and updates.

- **Disjoint Set Union (DSU):**
  For efficiently handling union and find operations.
  ```cpp
  struct DSU {
      std::vector<int> parent;
      DSU(int n) : parent(n+1) { for(int i=0;i<=n;i++) parent[i] = i; }
      int find_set(int x) { return parent[x] == x ? x : parent[x] = find_set(parent[x]); }
      void union_set(int x, int y) { parent[find_set(x)] = find_set(y); }
  };
  ```

- **Heaps (`std::priority_queue`):**
  Useful for efficiently retrieving the maximum or minimum element.

### 10. **Advanced Techniques**

- **Meet in the Middle:**
  Breaking problems into two halves to reduce time complexity.

- **Bitmasking and Enumeration:**
  Enumerating all subsets or combinations efficiently.

### **Best Practices**

- **Understand the Problem Thoroughly:**
  Carefully read and comprehend the problem constraints and requirements before jumping into coding.

- **Practice Code Implementation:**
  Regularly practice implementing various algorithms and data structures to build speed and accuracy.

- **Optimize and Test:**
  Continuously look for optimizations and thoroughly test your code against different cases to ensure correctness.

- **Stay Updated:**
  Keep abreast of new algorithms and techniques emerging in the competitive programming community.

By mastering these methods and leveraging C++'s powerful features, competitive programmers can efficiently tackle a wide array of challenging problems and excel in contests.
