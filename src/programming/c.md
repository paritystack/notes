# C Programming

## Overview

C is a general-purpose, procedural programming language developed by Dennis Ritchie at Bell Labs in 1972. It's widely used for system programming, embedded systems, operating systems (Unix/Linux), and applications requiring high performance and low-level memory access.

**Key Features:**
- **Low-level access to memory** via pointers
- **Efficient execution** with minimal runtime overhead
- **Portable** across different platforms
- **Rich library** of functions
- **Structured programming** with functions and modular code
- **Static typing** with compile-time type checking

## Basic Syntax

### Program Structure

```c
#include <stdio.h>     // Preprocessor directive
#include <stdlib.h>

// Function prototype
int add(int a, int b);

// Main function - entry point
int main(void) {
    printf("Hello, World!\n");

    int result = add(5, 3);
    printf("5 + 3 = %d\n", result);

    return 0;  // Return success code
}

// Function definition
int add(int a, int b) {
    return a + b;
}
```

### Comments

```c
// Single-line comment

/*
 * Multi-line comment
 * Spans multiple lines
 */
```

## Data Types

### Primitive Data Types

```c
// Integer types
char c = 'A';              // 1 byte: -128 to 127
unsigned char uc = 255;    // 1 byte: 0 to 255
short s = 32000;           // 2 bytes: -32,768 to 32,767
unsigned short us = 65000; // 2 bytes: 0 to 65,535
int i = 100000;            // 4 bytes: -2,147,483,648 to 2,147,483,647
unsigned int ui = 400000;  // 4 bytes: 0 to 4,294,967,295
long l = 1000000L;         // 4 or 8 bytes (platform-dependent)
unsigned long ul = 2000000UL;
long long ll = 9223372036854775807LL;  // 8 bytes

// Floating-point types
float f = 3.14f;           // 4 bytes, ~7 decimal digits precision
double d = 3.14159265359;  // 8 bytes, ~15 decimal digits precision
long double ld = 3.14159265358979323846L;  // 10-16 bytes

// Boolean (C99 and later)
#include <stdbool.h>
bool flag = true;          // true or false
```

### Size of Data Types

```c
#include <stdio.h>

int main(void) {
    printf("Size of char: %zu bytes\n", sizeof(char));
    printf("Size of int: %zu bytes\n", sizeof(int));
    printf("Size of float: %zu bytes\n", sizeof(float));
    printf("Size of double: %zu bytes\n", sizeof(double));
    printf("Size of pointer: %zu bytes\n", sizeof(void*));
    return 0;
}
```

## Variables and Constants

### Variable Declaration

```c
int x;               // Declaration
int y = 10;          // Declaration with initialization
int a, b, c;         // Multiple declarations
int m = 5, n = 10;   // Multiple with initialization

// Variable naming rules:
// - Must start with letter or underscore
// - Can contain letters, digits, underscores
// - Case-sensitive
// - Cannot use reserved keywords
```

### Constants

```c
// Using const keyword
const int MAX_SIZE = 100;
const double PI = 3.14159;

// Using #define preprocessor
#define BUFFER_SIZE 1024
#define TRUE 1
#define FALSE 0

// Enumeration constants
enum Color {
    RED,     // 0
    GREEN,   // 1
    BLUE     // 2
};

enum Status {
    SUCCESS = 0,
    ERROR = -1,
    PENDING = 1
};
```

## Operators

### Arithmetic Operators

```c
int a = 10, b = 3;
int sum = a + b;        // Addition: 13
int diff = a - b;       // Subtraction: 7
int prod = a * b;       // Multiplication: 30
int quot = a / b;       // Division: 3 (integer division)
int rem = a % b;        // Modulus: 1

// Increment/Decrement
int x = 5;
x++;    // Post-increment: x = 6
++x;    // Pre-increment: x = 7
x--;    // Post-decrement: x = 6
--x;    // Pre-decrement: x = 5
```

### Relational Operators

```c
int a = 5, b = 10;
int result;

result = (a == b);  // Equal to: 0 (false)
result = (a != b);  // Not equal: 1 (true)
result = (a > b);   // Greater than: 0
result = (a < b);   // Less than: 1
result = (a >= b);  // Greater than or equal: 0
result = (a <= b);  // Less than or equal: 1
```

### Logical Operators

```c
int a = 1, b = 0;

int and_result = a && b;  // Logical AND: 0
int or_result = a || b;   // Logical OR: 1
int not_result = !a;      // Logical NOT: 0
```

### Bitwise Operators

```c
unsigned int a = 5;   // 0101 in binary
unsigned int b = 3;   // 0011 in binary

unsigned int and = a & b;   // AND: 0001 (1)
unsigned int or = a | b;    // OR: 0111 (7)
unsigned int xor = a ^ b;   // XOR: 0110 (6)
unsigned int not = ~a;      // NOT: 1010 (complement)
unsigned int left = a << 1; // Left shift: 1010 (10)
unsigned int right = a >> 1;// Right shift: 0010 (2)
```

### Assignment Operators

```c
int x = 10;
x += 5;   // x = x + 5;  (15)
x -= 3;   // x = x - 3;  (12)
x *= 2;   // x = x * 2;  (24)
x /= 4;   // x = x / 4;  (6)
x %= 5;   // x = x % 5;  (1)
x &= 3;   // x = x & 3;
x |= 2;   // x = x | 2;
x ^= 1;   // x = x ^ 1;
x <<= 1;  // x = x << 1;
x >>= 1;  // x = x >> 1;
```

### Ternary Operator

```c
int a = 10, b = 20;
int max = (a > b) ? a : b;  // max = 20

// Equivalent to:
int max;
if (a > b) {
    max = a;
} else {
    max = b;
}
```

## Control Flow

### if-else Statements

```c
int age = 18;

if (age >= 18) {
    printf("Adult\n");
} else {
    printf("Minor\n");
}

// if-else if-else
int score = 85;
if (score >= 90) {
    printf("Grade: A\n");
} else if (score >= 80) {
    printf("Grade: B\n");
} else if (score >= 70) {
    printf("Grade: C\n");
} else {
    printf("Grade: F\n");
}

// Nested if
int x = 10, y = 20;
if (x > 0) {
    if (y > 0) {
        printf("Both positive\n");
    }
}
```

### switch Statement

```c
int day = 3;

switch (day) {
    case 1:
        printf("Monday\n");
        break;
    case 2:
        printf("Tuesday\n");
        break;
    case 3:
        printf("Wednesday\n");
        break;
    case 4:
        printf("Thursday\n");
        break;
    case 5:
        printf("Friday\n");
        break;
    case 6:
    case 7:
        printf("Weekend\n");
        break;
    default:
        printf("Invalid day\n");
        break;
}

// Switch with fall-through
char grade = 'B';
switch (grade) {
    case 'A':
    case 'B':
    case 'C':
        printf("Pass\n");
        break;
    case 'D':
    case 'F':
        printf("Fail\n");
        break;
    default:
        printf("Invalid grade\n");
}
```

### Loops

#### for Loop

```c
// Basic for loop
for (int i = 0; i < 10; i++) {
    printf("%d ", i);
}

// Nested for loop
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        printf("(%d, %d) ", i, j);
    }
    printf("\n");
}

// Multiple expressions
for (int i = 0, j = 10; i < 10; i++, j--) {
    printf("i=%d, j=%d\n", i, j);
}

// Infinite loop
for (;;) {
    // Loop forever (use break to exit)
    break;
}
```

#### while Loop

```c
int count = 0;
while (count < 5) {
    printf("%d ", count);
    count++;
}

// Reading input until condition
int num;
printf("Enter positive numbers (0 to stop): ");
while (scanf("%d", &num) == 1 && num != 0) {
    printf("You entered: %d\n", num);
}

// Infinite loop
while (1) {
    // Loop forever
    break;
}
```

#### do-while Loop

```c
int num;
do {
    printf("Enter a positive number: ");
    scanf("%d", &num);
} while (num <= 0);

// Executes at least once
int x = 10;
do {
    printf("x = %d\n", x);
    x++;
} while (x < 5);  // Condition false, but body executes once
```

### Loop Control Statements

```c
// break - exits the loop
for (int i = 0; i < 10; i++) {
    if (i == 5) break;
    printf("%d ", i);  // Prints: 0 1 2 3 4
}

// continue - skips to next iteration
for (int i = 0; i < 10; i++) {
    if (i % 2 == 0) continue;
    printf("%d ", i);  // Prints: 1 3 5 7 9
}

// goto - jumps to a label (use sparingly)
int i = 0;
start:
    printf("%d ", i);
    i++;
    if (i < 5) goto start;
```

## Functions

### Function Declaration and Definition

```c
// Function prototype (declaration)
int add(int a, int b);
void greet(void);
double calculate(int x, double y);

// Function definition
int add(int a, int b) {
    return a + b;
}

void greet(void) {
    printf("Hello!\n");
    // No return statement needed for void
}

double calculate(int x, double y) {
    return x * y;
}
```

### Function Parameters

```c
// Pass by value
void increment(int x) {
    x++;  // Only affects local copy
}

int main(void) {
    int num = 5;
    increment(num);
    printf("%d\n", num);  // Still 5
    return 0;
}

// Pass by reference (using pointers)
void increment_ref(int *x) {
    (*x)++;  // Modifies original value
}

int main(void) {
    int num = 5;
    increment_ref(&num);
    printf("%d\n", num);  // Now 6
    return 0;
}
```

### Return Values

```c
// Return single value
int square(int x) {
    return x * x;
}

// Return multiple values via pointers
void divide(int a, int b, int *quotient, int *remainder) {
    *quotient = a / b;
    *remainder = a % b;
}

int main(void) {
    int q, r;
    divide(10, 3, &q, &r);
    printf("10 / 3 = %d remainder %d\n", q, r);
    return 0;
}
```

### Variadic Functions

```c
#include <stdarg.h>

// Function with variable number of arguments
int sum(int count, ...) {
    va_list args;
    va_start(args, count);

    int total = 0;
    for (int i = 0; i < count; i++) {
        total += va_arg(args, int);
    }

    va_end(args);
    return total;
}

int main(void) {
    printf("Sum: %d\n", sum(3, 10, 20, 30));  // 60
    printf("Sum: %d\n", sum(5, 1, 2, 3, 4, 5));  // 15
    return 0;
}
```

### Recursive Functions

```c
// Factorial
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Fibonacci
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// Binary search (recursive)
int binary_search(int arr[], int left, int right, int target) {
    if (left > right) return -1;

    int mid = left + (right - left) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] > target) return binary_search(arr, left, mid - 1, target);
    return binary_search(arr, mid + 1, right, target);
}
```

## Arrays

### Array Declaration and Initialization

```c
// Declaration
int numbers[5];

// Declaration with initialization
int primes[5] = {2, 3, 5, 7, 11};

// Partial initialization (rest are 0)
int values[10] = {1, 2, 3};  // {1, 2, 3, 0, 0, 0, 0, 0, 0, 0}

// Size inferred from initializer
int data[] = {10, 20, 30, 40};  // Size: 4

// Zero-initialize all elements
int zeros[100] = {0};
```

### Accessing Array Elements

```c
int arr[5] = {10, 20, 30, 40, 50};

// Access elements
int first = arr[0];   // 10
int third = arr[2];   // 30

// Modify elements
arr[1] = 25;          // arr is now {10, 25, 30, 40, 50}

// Loop through array
for (int i = 0; i < 5; i++) {
    printf("%d ", arr[i]);
}
```

### Multi-dimensional Arrays

```c
// 2D array
int matrix[3][4] = {
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12}
};

// Access elements
int value = matrix[1][2];  // 7

// Loop through 2D array
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
        printf("%d ", matrix[i][j]);
    }
    printf("\n");
}

// 3D array
int cube[2][3][4];
```

### Arrays and Pointers

```c
int arr[5] = {10, 20, 30, 40, 50};

// Array name is a pointer to first element
int *ptr = arr;  // Same as &arr[0]

// Pointer arithmetic
printf("%d\n", *ptr);       // 10
printf("%d\n", *(ptr + 1)); // 20
printf("%d\n", *(ptr + 2)); // 30

// Equivalent notations
arr[2] == *(arr + 2) == *(ptr + 2) == ptr[2]  // All equal 30
```

### Passing Arrays to Functions

```c
// Array passed as pointer
void print_array(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Equivalent declaration
void print_array(int *arr, int size) {
    // Same as above
}

// 2D array
void print_matrix(int rows, int cols, int matrix[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}
```

## Pointers

### Pointer Basics

```c
int x = 10;
int *ptr = &x;     // ptr stores address of x

printf("Value of x: %d\n", x);           // 10
printf("Address of x: %p\n", (void*)&x); // Memory address
printf("Value of ptr: %p\n", (void*)ptr);// Same as &x
printf("Value at ptr: %d\n", *ptr);      // 10 (dereference)

// Modify through pointer
*ptr = 20;
printf("New value of x: %d\n", x);       // 20
```

### Pointer Arithmetic

```c
int arr[5] = {10, 20, 30, 40, 50};
int *ptr = arr;

printf("%d\n", *ptr);       // 10
printf("%d\n", *(ptr + 1)); // 20
printf("%d\n", *(ptr + 2)); // 30

ptr++;                      // Move to next element
printf("%d\n", *ptr);       // 20

ptr += 2;                   // Move 2 elements forward
printf("%d\n", *ptr);       // 40
```

### Pointer to Pointer

```c
int x = 10;
int *ptr1 = &x;
int **ptr2 = &ptr1;

printf("%d\n", **ptr2);  // 10

// Modify through double pointer
**ptr2 = 20;
printf("%d\n", x);       // 20
```

### Null Pointers

```c
int *ptr = NULL;  // Initialize to NULL

// Always check before dereferencing
if (ptr != NULL) {
    printf("%d\n", *ptr);
} else {
    printf("Pointer is NULL\n");
}
```

### Function Pointers

```c
// Function pointer declaration
int (*func_ptr)(int, int);

int add(int a, int b) {
    return a + b;
}

int multiply(int a, int b) {
    return a * b;
}

int main(void) {
    func_ptr = add;
    printf("10 + 5 = %d\n", func_ptr(10, 5));  // 15

    func_ptr = multiply;
    printf("10 * 5 = %d\n", func_ptr(10, 5));  // 50

    return 0;
}

// Array of function pointers
int (*operations[2])(int, int) = {add, multiply};
printf("Result: %d\n", operations[0](10, 5));  // 15
```

## Structures (Structs)

### Struct Declaration and Initialization

```c
// Define struct
struct Point {
    int x;
    int y;
};

// Create struct variable
struct Point p1;
p1.x = 10;
p1.y = 20;

// Initialize during declaration
struct Point p2 = {30, 40};

// Designated initializers (C99)
struct Point p3 = {.x = 50, .y = 60};
```

### Typedef with Structs

```c
typedef struct {
    char name[50];
    int age;
    float gpa;
} Student;

// Now can use Student instead of struct Student
Student s1 = {"Alice", 20, 3.8};
printf("Name: %s, Age: %d, GPA: %.2f\n", s1.name, s1.age, s1.gpa);
```

### Nested Structures

```c
typedef struct {
    int day;
    int month;
    int year;
} Date;

typedef struct {
    char name[50];
    Date birthdate;
    float salary;
} Employee;

Employee emp = {"John", {15, 8, 1990}, 50000.0};
printf("Name: %s\n", emp.name);
printf("Birthdate: %d/%d/%d\n", emp.birthdate.day,
       emp.birthdate.month, emp.birthdate.year);
```

### Pointers to Structures

```c
typedef struct {
    int x;
    int y;
} Point;

Point p1 = {10, 20};
Point *ptr = &p1;

// Access members through pointer
printf("x: %d, y: %d\n", (*ptr).x, (*ptr).y);

// Arrow operator (shorthand)
printf("x: %d, y: %d\n", ptr->x, ptr->y);
```

### Arrays of Structures

```c
typedef struct {
    char name[30];
    int age;
} Person;

Person people[3] = {
    {"Alice", 25},
    {"Bob", 30},
    {"Charlie", 35}
};

for (int i = 0; i < 3; i++) {
    printf("%s is %d years old\n", people[i].name, people[i].age);
}
```

## Unions and Enums

### Unions

```c
// Union: all members share same memory location
union Data {
    int i;
    float f;
    char c;
};

union Data data;
data.i = 10;
printf("i: %d\n", data.i);

data.f = 3.14;  // Overwrites i
printf("f: %.2f\n", data.f);
printf("i: %d\n", data.i);  // Corrupted

printf("Size of union: %zu\n", sizeof(union Data));  // Size of largest member
```

### Enumerations

```c
// Define enum
enum Day {
    MONDAY,    // 0
    TUESDAY,   // 1
    WEDNESDAY, // 2
    THURSDAY,  // 3
    FRIDAY,    // 4
    SATURDAY,  // 5
    SUNDAY     // 6
};

enum Day today = WEDNESDAY;

// Custom values
enum Status {
    SUCCESS = 0,
    ERROR = -1,
    PENDING = 1,
    TIMEOUT = 2
};

// Typedef with enum
typedef enum {
    RED,
    GREEN,
    BLUE
} Color;

Color favorite = BLUE;
```

## File I/O

### Opening and Closing Files

```c
#include <stdio.h>

FILE *file = fopen("data.txt", "r");  // Open for reading
if (file == NULL) {
    perror("Error opening file");
    return 1;
}

// Use file...

fclose(file);  // Always close when done
```

**File Modes:**
- `"r"` - Read (file must exist)
- `"w"` - Write (creates new or truncates existing)
- `"a"` - Append (creates new or appends to existing)
- `"r+"` - Read and write (file must exist)
- `"w+"` - Read and write (creates new or truncates)
- `"a+"` - Read and append

### Writing to Files

```c
// fprintf - formatted output
FILE *file = fopen("output.txt", "w");
fprintf(file, "Hello, %s!\n", "World");
fprintf(file, "Number: %d\n", 42);
fclose(file);

// fputs - write string
FILE *file = fopen("output.txt", "w");
fputs("Line 1\n", file);
fputs("Line 2\n", file);
fclose(file);

// fwrite - binary write
int numbers[] = {1, 2, 3, 4, 5};
FILE *file = fopen("data.bin", "wb");
fwrite(numbers, sizeof(int), 5, file);
fclose(file);
```

### Reading from Files

```c
// fscanf - formatted input
FILE *file = fopen("input.txt", "r");
int num;
char str[50];
fscanf(file, "%d %s", &num, str);
fclose(file);

// fgets - read line
FILE *file = fopen("input.txt", "r");
char line[100];
while (fgets(line, sizeof(line), file) != NULL) {
    printf("%s", line);
}
fclose(file);

// fread - binary read
int numbers[5];
FILE *file = fopen("data.bin", "rb");
fread(numbers, sizeof(int), 5, file);
fclose(file);

// fgetc - read character
FILE *file = fopen("input.txt", "r");
int ch;
while ((ch = fgetc(file)) != EOF) {
    putchar(ch);
}
fclose(file);
```

### File Position Functions

```c
FILE *file = fopen("data.txt", "r");

// ftell - get current position
long pos = ftell(file);

// fseek - set position
fseek(file, 0, SEEK_SET);  // Beginning of file
fseek(file, 0, SEEK_END);  // End of file
fseek(file, 10, SEEK_CUR); // 10 bytes from current position

// rewind - reset to beginning
rewind(file);

fclose(file);
```

### File Error Checking

```c
FILE *file = fopen("data.txt", "r");
if (file == NULL) {
    perror("fopen");
    return 1;
}

// Check for errors
if (ferror(file)) {
    fprintf(stderr, "Error reading file\n");
}

// Check for end of file
if (feof(file)) {
    printf("End of file reached\n");
}

fclose(file);
```

## Preprocessor Directives

### #include Directive

```c
#include <stdio.h>     // System header
#include <stdlib.h>
#include <string.h>

#include "myheader.h"  // User-defined header
```

### #define Directive

```c
// Constants
#define PI 3.14159
#define MAX_SIZE 1000
#define BUFFER_LEN 256

// Macros
#define SQUARE(x) ((x) * (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Multi-line macro
#define SWAP(a, b, type) do { \
    type temp = a;            \
    a = b;                    \
    b = temp;                 \
} while(0)

// Usage
int result = SQUARE(5);  // 25
int max_val = MAX(10, 20);  // 20
```

### Conditional Compilation

```c
#define DEBUG 1

#ifdef DEBUG
    printf("Debug mode enabled\n");
#endif

#ifndef RELEASE
    printf("Not in release mode\n");
#endif

#if DEBUG == 1
    printf("Debug level 1\n");
#elif DEBUG == 2
    printf("Debug level 2\n");
#else
    printf("Debug disabled\n");
#endif

// Prevent multiple inclusion
#ifndef MYHEADER_H
#define MYHEADER_H

// Header contents...

#endif  // MYHEADER_H
```

### Predefined Macros

```c
printf("File: %s\n", __FILE__);      // Current filename
printf("Line: %d\n", __LINE__);      // Current line number
printf("Date: %s\n", __DATE__);      // Compilation date
printf("Time: %s\n", __TIME__);      // Compilation time
printf("Function: %s\n", __func__);  // Current function name (C99)
```

### #undef and #pragma

```c
// Undefine a macro
#define TEMP 100
#undef TEMP

// Compiler-specific directives
#pragma once  // Alternative to include guards (non-standard)
#pragma pack(1)  // Structure packing
```

## Common Patterns

### Error Handling Pattern

```c
int process_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("fopen");
        return -1;
    }

    char *buffer = malloc(1024);
    if (buffer == NULL) {
        fclose(file);
        perror("malloc");
        return -1;
    }

    // Process file...

    // Cleanup
    free(buffer);
    fclose(file);
    return 0;
}
```

### Generic Swap Function

```c
void swap(void *a, void *b, size_t size) {
    unsigned char *p = a;
    unsigned char *q = b;
    unsigned char temp;

    for (size_t i = 0; i < size; i++) {
        temp = p[i];
        p[i] = q[i];
        q[i] = temp;
    }
}

// Usage
int x = 10, y = 20;
swap(&x, &y, sizeof(int));
printf("x=%d, y=%d\n", x, y);  // x=20, y=10
```

### Linked List Implementation

```c
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// Insert at beginning
Node* insert_front(Node *head, int data) {
    Node *new_node = malloc(sizeof(Node));
    if (new_node == NULL) return head;

    new_node->data = data;
    new_node->next = head;
    return new_node;
}

// Print list
void print_list(Node *head) {
    Node *current = head;
    while (current != NULL) {
        printf("%d -> ", current->data);
        current = current->next;
    }
    printf("NULL\n");
}

// Free list
void free_list(Node *head) {
    Node *current = head;
    while (current != NULL) {
        Node *temp = current;
        current = current->next;
        free(temp);
    }
}
```

### Command Line Arguments

```c
int main(int argc, char *argv[]) {
    printf("Program name: %s\n", argv[0]);
    printf("Number of arguments: %d\n", argc - 1);

    for (int i = 1; i < argc; i++) {
        printf("Argument %d: %s\n", i, argv[i]);
    }

    return 0;
}

// Run: ./program arg1 arg2 arg3
// Output:
// Program name: ./program
// Number of arguments: 3
// Argument 1: arg1
// Argument 2: arg2
// Argument 3: arg3
```

## Best Practices

### Code Organization

```c
// Use meaningful names
int calculate_average(int *scores, int count);  // Good
int calc(int *a, int n);                        // Avoid

// Use constants instead of magic numbers
#define MAX_STUDENTS 100
int students[MAX_STUDENTS];  // Good

int students[100];  // Avoid

// Group related code
typedef struct {
    char name[50];
    int age;
} Person;

Person create_person(const char *name, int age);
void print_person(const Person *p);
void free_person(Person *p);
```

### Memory Management

```c
// Always check malloc return value
int *ptr = malloc(sizeof(int) * 100);
if (ptr == NULL) {
    fprintf(stderr, "Memory allocation failed\n");
    return -1;
}

// Always free allocated memory
free(ptr);
ptr = NULL;  // Prevent dangling pointer

// Avoid memory leaks
void bad_function() {
    int *data = malloc(sizeof(int) * 100);
    if (some_error) {
        return;  // LEAK! Forgot to free
    }
    free(data);
}

void good_function() {
    int *data = malloc(sizeof(int) * 100);
    if (data == NULL) return;

    if (some_error) {
        free(data);  // Clean up before return
        return;
    }

    free(data);
}
```

### Buffer Safety

```c
// Use strncpy instead of strcpy
char dest[20];
strncpy(dest, source, sizeof(dest) - 1);
dest[sizeof(dest) - 1] = '\0';  // Ensure null termination

// Use snprintf instead of sprintf
char buffer[50];
snprintf(buffer, sizeof(buffer), "Value: %d", value);

// Check array bounds
for (int i = 0; i < array_size; i++) {
    // Safe access
}
```

### Function Design

```c
// Use const for read-only parameters
int calculate_sum(const int *arr, int size);

// Return error codes
int read_file(const char *filename, char **buffer) {
    if (filename == NULL || buffer == NULL) {
        return -1;  // Invalid parameters
    }

    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        return -2;  // File open error
    }

    // Success
    return 0;
}

// Use header guards
// myheader.h
#ifndef MYHEADER_H
#define MYHEADER_H

// Declarations...

#endif
```

### Compilation Flags

```bash
# Enable warnings
gcc -Wall -Wextra -Werror program.c -o program

# Debug symbols
gcc -g program.c -o program

# Optimization
gcc -O2 program.c -o program

# C standard
gcc -std=c11 program.c -o program

# Combine flags
gcc -Wall -Wextra -O2 -std=c11 program.c -o program
```

## Difference Between Different Const Pointers

In C programming, pointers can be declared with the `const` qualifier in different ways, leading to different types of constant pointers. Understanding these differences is crucial for writing correct and efficient code.

1. **Pointer to a Constant Variable:**
   A pointer to a constant variable means that the value being pointed to cannot be changed through the pointer, but the pointer itself can be changed to point to another variable.
   ```c
   const int *ptr;
   int a = 10;
   int b = 20;
   ptr = &a; // Valid
   *ptr = 30; // Invalid, cannot change the value of 'a' through ptr
   ptr = &b; // Valid, can change the pointer to point to 'b'
   ```

2. **Constant Pointer to a Variable:**
   A constant pointer to a variable means that the pointer itself cannot be changed to point to another variable, but the value being pointed to can be changed.
   ```c
   int *const ptr = &a;
   int a = 10;
   int b = 20;
   ptr = &b; // Invalid, cannot change the pointer to point to 'b'
   *ptr = 30; // Valid, can change the value of 'a' through ptr
   ```

3. **Constant Pointer to a Constant Variable:**
   A constant pointer to a constant variable means that neither the pointer can be changed to point to another variable nor the value being pointed to can be changed.
   ```c
   const int *const ptr = &a;
   int a = 10;
   int b = 20;
   ptr = &b; // Invalid, cannot change the pointer to point to 'b'
   *ptr = 30; // Invalid, cannot change the value of 'a' through ptr
   ```

These different types of constant pointers provide various levels of protection and control over the data and pointers in your program, helping to prevent unintended modifications and ensuring code reliability.

## Dynamic Memory Allocation

Dynamic memory allocation allows you to allocate memory at runtime instead of compile time. This is essential for creating data structures of variable size.

### Memory Layout

```
Stack (grows down)   |  Local variables, function parameters
                     |
                     V
=====================|======================  <- Stack limit
                     ^
                     |
                     |  Free memory
                     |
                     V
=====================|======================  <- Heap limit
Heap (grows up)      |  malloc, calloc, realloc allocations
                     ^
```

### malloc() - Memory Allocation

Allocates memory and returns a void pointer:

```c
#include <stdlib.h>

// Allocate memory for single integer
int *ptr = (int *)malloc(sizeof(int));
if (ptr == NULL) {
    printf("Memory allocation failed\n");
    return 1;
}
*ptr = 42;
printf("Value: %d\n", *ptr);
free(ptr);
ptr = NULL;  // Good practice: set to NULL after free

// Allocate memory for array
int *arr = (int *)malloc(10 * sizeof(int));
if (arr == NULL) {
    printf("Memory allocation failed\n");
    return 1;
}
arr[0] = 100;
arr[9] = 999;
free(arr);
arr = NULL;
```

### calloc() - Contiguous Memory Allocation

Allocates memory and initializes all bytes to zero:

```c
#include <stdlib.h>

// calloc(number_of_elements, size_of_each_element)
int *arr = (int *)calloc(10, sizeof(int));  // 10 integers, all initialized to 0
if (arr == NULL) {
    printf("Memory allocation failed\n");
    return 1;
}

for (int i = 0; i < 10; i++) {
    printf("%d ", arr[i]);  // Prints: 0 0 0 0 0 0 0 0 0 0
}

free(arr);
arr = NULL;
```

### realloc() - Resize Memory

Resizes previously allocated memory block:

```c
#include <stdlib.h>

int *arr = (int *)malloc(5 * sizeof(int));
for (int i = 0; i < 5; i++) arr[i] = i;

// Resize to 10 integers
int *new_arr = (int *)realloc(arr, 10 * sizeof(int));
if (new_arr == NULL) {
    printf("Reallocation failed\n");
    free(arr);  // Original block still exists
    return 1;
}

arr = new_arr;
for (int i = 5; i < 10; i++) arr[i] = i;

free(arr);
arr = NULL;
```

### free() - Deallocate Memory

Deallocates previously allocated memory:

```c
#include <stdlib.h>

int *ptr = (int *)malloc(sizeof(int));
*ptr = 42;

// When done, free the memory
free(ptr);

// IMPORTANT: Set to NULL to avoid dangling pointer
ptr = NULL;
```

### Memory Allocation Pattern (Safe)

```c
#include <stdlib.h>
#include <stdio.h>

int main(void) {
    // 1. Declare pointer
    int *ptr;

    // 2. Allocate memory with error checking
    ptr = (int *)malloc(sizeof(int));
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 3. Use the memory
    *ptr = 100;
    printf("Value: %d\n", *ptr);

    // 4. Free the memory
    free(ptr);

    // 5. Set to NULL (avoid dangling pointer)
    ptr = NULL;

    return 0;
}
```

### Dynamic Array Implementation

```c
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    int *data;
    int size;
    int capacity;
} DynamicArray;

// Create dynamic array
DynamicArray* array_create(int initial_capacity) {
    DynamicArray *arr = (DynamicArray *)malloc(sizeof(DynamicArray));
    if (arr == NULL) return NULL;

    arr->data = (int *)malloc(initial_capacity * sizeof(int));
    if (arr->data == NULL) {
        free(arr);
        return NULL;
    }

    arr->size = 0;
    arr->capacity = initial_capacity;
    return arr;
}

// Add element to array
int array_push(DynamicArray *arr, int value) {
    if (arr->size == arr->capacity) {
        // Resize: double the capacity
        int new_capacity = arr->capacity * 2;
        int *new_data = (int *)realloc(arr->data, new_capacity * sizeof(int));
        if (new_data == NULL) return -1;

        arr->data = new_data;
        arr->capacity = new_capacity;
    }

    arr->data[arr->size++] = value;
    return 0;
}

// Get element from array
int array_get(DynamicArray *arr, int index) {
    if (index < 0 || index >= arr->size) {
        fprintf(stderr, "Index out of bounds\n");
        return -1;
    }
    return arr->data[index];
}

// Free array
void array_free(DynamicArray *arr) {
    if (arr == NULL) return;
    free(arr->data);
    free(arr);
}

// Usage
int main(void) {
    DynamicArray *arr = array_create(10);
    if (arr == NULL) {
        fprintf(stderr, "Failed to create array\n");
        return 1;
    }

    for (int i = 0; i < 20; i++) {
        array_push(arr, i * 10);
    }

    for (int i = 0; i < arr->size; i++) {
        printf("%d ", array_get(arr, i));
    }

    array_free(arr);
    return 0;
}
```

### Common Memory Errors

#### 1. Memory Leak (forgot to free)
```c
void memory_leak(void) {
    int *ptr = (int *)malloc(sizeof(int));
    *ptr = 42;
    // Missing: free(ptr);
    // Memory is lost when function exits
}
```

#### 2. Double Free
```c
int *ptr = (int *)malloc(sizeof(int));
free(ptr);
free(ptr);  // ERROR: Undefined behavior!
```

#### 3. Use After Free (Dangling Pointer)
```c
int *ptr = (int *)malloc(sizeof(int));
*ptr = 42;
free(ptr);
printf("%d\n", *ptr);  // ERROR: ptr points to freed memory!
ptr = NULL;  // Should do this after free
```

#### 4. Buffer Overflow
```c
char *str = (char *)malloc(5);
strcpy(str, "Hello World");  // ERROR: Buffer overflow!
                             // "Hello World" needs 12 bytes, only allocated 5
free(str);
```

#### 5. Null Pointer Dereference
```c
int *ptr = (int *)malloc(sizeof(int));
// Allocation failed
if (ptr == NULL) {
    *ptr = 42;  // ERROR: Dereferencing NULL!
}
```

### Best Practices for Dynamic Memory

```c
// 1. Always check if malloc/calloc/realloc succeeded
int *ptr = (int *)malloc(sizeof(int));
if (ptr == NULL) {
    // Handle error
    return -1;
}

// 2. Use sizeof for type safety
int *arr = (int *)malloc(100 * sizeof(int));  // Good
int *arr = (int *)malloc(100 * 4);            // Avoid - hardcoded size

// 3. Free in reverse order of allocation
void *p1 = malloc(100);
void *p2 = malloc(200);
void *p3 = malloc(300);

free(p3);
free(p2);
free(p1);

// 4. Set pointer to NULL after free
free(ptr);
ptr = NULL;

// 5. Avoid memory leaks - create cleanup paths
FILE *file = fopen("data.txt", "r");
int *data = (int *)malloc(1000 * sizeof(int));

if (file == NULL) {
    free(data);  // Clean up before returning
    return -1;
}

// Process file
if (some_error) {
    fclose(file);
    free(data);  // Clean up before returning
    return -1;
}

fclose(file);
free(data);
return 0;

// 6. Use wrapper functions for consistency
void* safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "malloc failed: requested %zu bytes\n", size);
        exit(1);  // Or handle error differently
    }
    return ptr;
}

int *arr = (int *)safe_malloc(100 * sizeof(int));
```

### Memory Leak Detection Tools

```bash
# Valgrind - memory error detector
valgrind --leak-check=full --show-leak-kinds=all ./program

# AddressSanitizer (GCC/Clang)
gcc -fsanitize=address -g program.c -o program
./program

# Dr. Memory (Windows)
drmemory -leaks_only -- program.exe
```

### Comparison of Allocation Functions

| Function | Initialization | Returns NULL on Fail | Use Case |
|----------|----------------|----------------------|----------|
| **malloc** | No (garbage values) | Yes | When you'll initialize manually |
| **calloc** | Yes (all zeros) | Yes | When you need zeroed memory |
| **realloc** | Preserves existing | Yes | When resizing allocations |

### Memory Allocation Time Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| **malloc/calloc** | O(1) amortized | O(1) |
| **free** | O(1) amortized | O(1) |
| **realloc** | O(n) | O(n) |

## Commonly Used String Library Functions

The C standard library provides a set of functions for manipulating strings. Here are some commonly used string functions:

1. **strlen** - Calculate the length of a string:
   ```c
   #include <string.h>
   size_t length = strlen("example");
   ```

2. **strcpy** - Copy a string:
   ```c
   #include <string.h>
   char dest[20];
   strcpy(dest, "source");
   ```

3. **strncpy** - Copy a specified number of characters from a string:
   ```c
   #include <string.h>
   char dest[20];
   strncpy(dest, "source", 5);
   ```

4. **strcat** - Concatenate two strings:
   ```c
   #include <string.h>
   char dest[20] = "Hello, ";
   strcat(dest, "World!");
   ```

5. **strncat** - Concatenate a specified number of characters from one string to another:
   ```c
   #include <string.h>
   char dest[20] = "Hello, ";
   strncat(dest, "World!", 3);
   ```

6. **strcmp** - Compare two strings:
   ```c
   #include <string.h>
   int result = strcmp("string1", "string2");
   ```

7. **strncmp** - Compare a specified number of characters from two strings:
   ```c
   #include <string.h>
   int result = strncmp("string1", "string2", 5);
   ```

8. **strchr** - Find the first occurrence of a character in a string:
   ```c
   #include <string.h>
   char *ptr = strchr("example", 'a');
   ```

9. **strrchr** - Find the last occurrence of a character in a string:
   ```c
   #include <string.h>
   char *ptr = strrchr("example", 'e');
   ```

10. **strstr** - Find the first occurrence of a substring in a string:
    ```c
    #include <string.h>
    char *ptr = strstr("example", "amp");
    ```

These functions cover a variety of common use cases for string manipulation in C, making them essential tools for C programmers.

### Variants of `printf` and `scanf`

The `printf` and `scanf` functions are commonly used for input and output in C. There are several variants of these functions that provide additional functionality.

#### `printf` Variants

1. **`printf`** - Print formatted output to the standard output:
   ```c
   #include <stdio.h>
   printf("Hello, %s!\n", "World");
   ```

2. **`fprintf`** - Print formatted output to a file:
   ```c
   #include <stdio.h>
   FILE *file = fopen("output.txt", "w");
   fprintf(file, "Hello, %s!\n", "World");
   fclose(file);
   ```

3. **`sprintf`** - Print formatted output to a string:
   ```c
   #include <stdio.h>
   char buffer[50];
   sprintf(buffer, "Hello, %s!", "World");
   ```

4. **`snprintf`** - Print formatted output to a string with a limit on the number of characters:
   ```c
   #include <stdio.h>
   char buffer[50];
   snprintf(buffer, sizeof(buffer), "Hello, %s!", "World");
   ```

5. **`vprintf`** - Print formatted output using a `va_list`:
   ```c
   #include <stdio.h>
   #include <stdarg.h>
   void my_vprintf(const char *format, ...) {
       va_list args;
       va_start(args, format);
       vprintf(format, args);
       va_end(args);
   }
   ```

6. **`vfprintf`** - Print formatted output to a file using a `va_list`:
   ```c
   #include <stdio.h>
   #include <stdarg.h>
   void my_vfprintf(FILE *file, const char *format, ...) {
       va_list args;
       va_start(args, format);
       vfprintf(file, format, args);
       va_end(args);
   }
   ```

7. **`vsprintf`** - Print formatted output to a string using a `va_list`:
   ```c
   #include <stdio.h>
   #include <stdarg.h>
   void my_vsprintf(char *buffer, const char *format, ...) {
       va_list args;
       va_start(args, format);
       vsprintf(buffer, format, args);
       va_end(args);
   }
   ```

8. **`vsnprintf`** - Print formatted output to a string with a limit on the number of characters using a `va_list`:
   ```c
   #include <stdio.h>
   #include <stdarg.h>
   void my_vsnprintf(char *buffer, size_t size, const char *format, ...) {
       va_list args;
       va_start(args, format);
       vsnprintf(buffer, size, format, args);
       va_end(args);
   }
   ```

#### `scanf` Variants

1. **`scanf`** - Read formatted input from the standard input:
   ```c
   #include <stdio.h>
   int value;
   scanf("%d", &value);
   ```

2. **`fscanf`** - Read formatted input from a file:
   ```c
   #include <stdio.h>
   FILE *file = fopen("input.txt", "r");
   int value;
   fscanf(file, "%d", &value);
   fclose(file);
   ```

3. **`sscanf`** - Read formatted input from a string:
   ```c
   #include <stdio.h>
   const char *str = "123";
   int value;
   sscanf(str, "%d", &value);
   ```

4. **`vscanf`** - Read formatted input using a `va_list`:
   ```c
   #include <stdio.h>
   #include <stdarg.h>
   void my_vscanf(const char *format, ...) {
       va_list args;
       va_start(args, format);
       vscanf(format, args);
       va_end(args);
   }
   ```

5. **`vfscanf`** - Read formatted input from a file using a `va_list`:
   ```c
   #include <stdio.h>
   #include <stdarg.h>
   void my_vfscanf(FILE *file, const char *format, ...) {
       va_list args;
       va_start(args, format);
       vfscanf(file, format, args);
       va_end(args);
   }
   ```

6. **`vsscanf`** - Read formatted input from a string using a `va_list`:
   ```c
   #include <stdio.h>
   #include <stdarg.h>
   void my_vsscanf(const char *str, const char *format, ...) {
       va_list args;
       va_start(args, format);
       vsscanf(str, format, args);
       va_end(args);
   }
   ```

These variants of `printf` and `scanf` provide flexibility for different input and output scenarios in C programming.
