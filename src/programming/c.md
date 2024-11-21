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
