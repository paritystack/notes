# C Programming


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
