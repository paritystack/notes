# Memory Management

## Table of Contents
- [Memory Fundamentals](#memory-fundamentals)
  - [Stack vs Heap Allocation](#stack-vs-heap-allocation)
  - [Memory Layout](#memory-layout)
  - [Virtual Memory](#virtual-memory)
  - [Memory Alignment](#memory-alignment)
  - [Fragmentation](#fragmentation)
- [Allocation Strategies](#allocation-strategies)
  - [Static Allocation](#static-allocation)
  - [Stack Allocation](#stack-allocation)
  - [Heap Allocation](#heap-allocation)
  - [Memory Pools](#memory-pools)
  - [Arena Allocators](#arena-allocators)
- [Garbage Collection](#garbage-collection)
  - [Reference Counting](#reference-counting)
  - [Mark and Sweep](#mark-and-sweep)
  - [Generational GC](#generational-gc)
  - [Tri-Color Marking](#tri-color-marking)
  - [GC Tuning](#gc-tuning)
  - [GC Pauses](#gc-pauses)
- [Manual Memory Management](#manual-memory-management)
  - [malloc/free in C](#mallocfree-in-c)
  - [new/delete in C++](#newdelete-in-c)
  - [Memory Leak Detection](#memory-leak-detection)
  - [Use-After-Free Bugs](#use-after-free-bugs)
  - [Double-Free Errors](#double-free-errors)
- [Smart Pointers (C++)](#smart-pointers-c)
  - [unique_ptr](#unique_ptr)
  - [shared_ptr](#shared_ptr)
  - [weak_ptr](#weak_ptr)
  - [RAII Pattern](#raii-pattern)
- [Language-Specific Memory Management](#language-specific-memory-management)
  - [Python](#python)
  - [JavaScript](#javascript)
  - [Go](#go)
  - [Rust](#rust)
  - [Java](#java)
- [Memory Profiling](#memory-profiling)
  - [Profiling Tools](#profiling-tools)
  - [Memory Leak Detection Tools](#memory-leak-detection-tools)
  - [Heap Profiling](#heap-profiling)
- [Performance Optimization](#performance-optimization)
  - [Cache-Friendly Data Structures](#cache-friendly-data-structures)
  - [Memory Access Patterns](#memory-access-patterns)
  - [Copy-on-Write](#copy-on-write)
  - [Memory-Mapped Files](#memory-mapped-files)
- [Common Pitfalls and Best Practices](#common-pitfalls-and-best-practices)

---

## Memory Fundamentals

### Stack vs Heap Allocation

Memory in programs is primarily divided into two main areas: the stack and the heap. Understanding the differences is crucial for writing efficient and correct code.

#### The Stack

**Characteristics:**
- **Fast allocation/deallocation**: Push/pop operations (O(1))
- **Automatic management**: Variables automatically cleaned up when out of scope
- **Limited size**: Typically 1-8 MB (platform-dependent)
- **LIFO structure**: Last In, First Out
- **Thread-local**: Each thread has its own stack
- **Contiguous memory**: Sequential allocation

**What goes on the stack:**
- Local variables
- Function parameters
- Return addresses
- Function call frames

**Example in C:**
```c
void function() {
    int x = 10;           // Allocated on stack
    char buffer[100];     // Allocated on stack
    double y = 3.14;      // Allocated on stack
}  // All variables automatically destroyed here
```

**Stack Frame Structure:**
```
High Address
+------------------+
| Previous Frame   |
+------------------+
| Return Address   |
+------------------+
| Saved Registers  |
+------------------+
| Local Variables  |
+------------------+
| Arguments        |
+------------------+ <- Stack Pointer (SP)
Low Address
```

**Advantages:**
- Very fast allocation (just move stack pointer)
- No fragmentation
- Automatic cleanup
- Cache-friendly (locality of reference)

**Disadvantages:**
- Limited size (stack overflow risk)
- Variables destroyed when function returns
- Size must be known at compile time

#### The Heap

**Characteristics:**
- **Slower allocation/deallocation**: Requires bookkeeping
- **Manual or GC management**: Must explicitly free or use garbage collection
- **Large size**: Limited by available system memory
- **Flexible structure**: Can allocate any size at runtime
- **Shared**: Accessible by all threads (requires synchronization)
- **Fragmented**: Non-contiguous allocations

**What goes on the heap:**
- Dynamically allocated objects
- Large data structures
- Objects with unknown size at compile time
- Objects that need to outlive their scope

**Example in C:**
```c
void function() {
    int* ptr = malloc(sizeof(int) * 100);  // Allocated on heap
    // Use ptr...
    free(ptr);  // Must manually free
}
```

**Heap Structure:**
```
+------------------+
| Free Block       |
+------------------+
| Allocated Block  |
+------------------+
| Free Block       |
+------------------+
| Allocated Block  |
+------------------+
```

**Advantages:**
- Large size available
- Variables persist beyond function scope
- Runtime-sized allocations
- Flexible lifetime control

**Disadvantages:**
- Slower allocation
- Manual management (C/C++) or GC overhead
- Fragmentation issues
- Potential for memory leaks

#### Comparison Table

| Feature | Stack | Heap |
|---------|-------|------|
| Speed | Very fast (nanoseconds) | Slower (microseconds) |
| Size | Limited (1-8 MB) | Large (GB+) |
| Management | Automatic | Manual/GC |
| Lifetime | Function scope | Explicit control |
| Fragmentation | None | Possible |
| Thread-safety | Thread-local | Requires sync |
| Access pattern | Sequential | Random |

#### When to Use Each

**Use Stack:**
- Small, fixed-size data
- Short-lived variables
- When you need maximum speed
- When automatic cleanup is desired

**Use Heap:**
- Large data structures
- Data that outlives function scope
- Runtime-sized allocations
- Shared data between threads

### Memory Layout

Understanding how a program's memory is organized is essential for debugging and optimization.

#### Typical Memory Layout (32-bit/64-bit systems)

```
High Address (0xFFFFFFFF / 0xFFFFFFFFFFFFFFFF)
+------------------------+
|    Kernel Space        |
|  (OS, system calls)    |
+------------------------+ <- 0xC0000000 (varies)
|        Stack           |
|    (grows down)        |
|          ↓             |
+------------------------+
|         ...            |
|     (unmapped)         |
|         ...            |
+------------------------+
|          ↑             |
|    (grows up)          |
|         Heap           |
+------------------------+
|    BSS Segment         |
|  (uninitialized data)  |
+------------------------+
|    Data Segment        |
|  (initialized data)    |
+------------------------+
|    Text Segment        |
|   (code/instructions)  |
+------------------------+ <- 0x08048000 (typical)
|       Reserved         |
+------------------------+
Low Address (0x00000000)
```

#### Segment Details

**1. Text Segment (Code Segment)**
- Contains executable instructions
- Read-only and shareable
- Fixed size determined at compile time
- Contains program code and constants

```c
// This function's machine code goes in text segment
int add(int a, int b) {
    return a + b;
}

// String literal in text segment (read-only)
const char* msg = "Hello, World!";
```

**2. Data Segment (Initialized Data)**
- Contains global and static variables with initial values
- Read-write
- Fixed size

```c
// Goes in data segment
int global_initialized = 42;
static int static_initialized = 100;

void function() {
    static int func_static = 5;  // Also in data segment
}
```

**3. BSS Segment (Block Started by Symbol)**
- Contains uninitialized global and static variables
- Automatically zeroed by OS
- Doesn't take space in executable file (just a marker)

```c
// Goes in BSS segment
int global_uninitialized;
static int static_uninitialized;

void function() {
    static int func_static;  // Also in BSS
}
```

**Why separate BSS from Data?**
- Reduces executable file size
- No need to store zeros in the binary
- OS zeros memory pages when loading

**4. Heap**
- Dynamic memory allocation
- Grows upward (toward higher addresses)
- Managed by allocators (malloc/new)
- Shared by all threads

**5. Stack**
- Local variables and function calls
- Grows downward (toward lower addresses)
- Each thread has its own stack
- Limited size (configurable)

**6. Memory-Mapped Region**
- Shared libraries
- Memory-mapped files
- Between heap and stack

#### Example Program Memory

```c
#include <stdio.h>
#include <stdlib.h>

// BSS segment
int global_uninit;

// Data segment
int global_init = 42;

// Text segment
int add(int a, int b) {
    return a + b;
}

int main() {
    // Stack
    int stack_var = 10;

    // Heap
    int* heap_var = malloc(sizeof(int));
    *heap_var = 20;

    // Text segment (string literal)
    char* str = "Hello";

    printf("Stack var address: %p\n", (void*)&stack_var);
    printf("Heap var address: %p\n", (void*)heap_var);
    printf("Global init address: %p\n", (void*)&global_init);
    printf("Global uninit address: %p\n", (void*)&global_uninit);
    printf("Function address: %p\n", (void*)add);
    printf("String literal address: %p\n", (void*)str);

    free(heap_var);
    return 0;
}
```

**Output (example on Linux x86-64):**
```
Stack var address: 0x7ffd1234abcd
Heap var address: 0x55e4d789ef00
Global init address: 0x55e4d6789010
Global uninit address: 0x55e4d6789020
Function address: 0x55e4d6789140
String literal address: 0x55e4d6789200
```

Notice the pattern:
- Stack: High address
- Heap: Medium address
- Global data: Lower address
- Code/strings: Lowest address

#### Inspecting Memory Layout

**Linux:**
```bash
# View process memory map
cat /proc/<pid>/maps

# Example output:
# 00400000-00401000 r-xp   text segment
# 00601000-00602000 rw-p   data segment
# 00602000-00623000 rw-p   heap
# 7fff12340000-7fff12361000 rw-p   stack
```

**Using size command:**
```bash
$ size a.out
   text    data     bss     dec     hex filename
   1234     456     100    1790     6fe a.out
```

### Virtual Memory

Virtual memory is an abstraction that provides each process with the illusion of having its own private memory space.

#### Key Concepts

**1. Virtual Address Space**
- Each process has its own virtual address space
- Typically 2^32 bytes (4 GB) on 32-bit systems
- Typically 2^48 bytes (256 TB) on 64-bit systems
- Isolated from other processes

**2. Physical Memory**
- Actual RAM installed in the system
- Shared among all processes
- Much smaller than total virtual memory

**3. Address Translation**

```
Virtual Address → MMU → Physical Address
```

**Components:**
- **MMU (Memory Management Unit)**: Hardware that translates virtual to physical addresses
- **Page Table**: Maps virtual pages to physical frames
- **TLB (Translation Lookaside Buffer)**: Cache for page table entries

#### Paging

Memory is divided into fixed-size blocks:
- **Pages**: Fixed-size blocks in virtual memory (typically 4 KB)
- **Frames**: Fixed-size blocks in physical memory (same size as pages)

**Page Table Structure:**
```
Virtual Page Number (VPN) → Page Table → Physical Frame Number (PFN)
```

**Example:**
```
Virtual Address: 0x00403004
Page Size: 4096 bytes (4 KB)

VPN = 0x00403004 / 4096 = 0x403
Offset = 0x00403004 % 4096 = 0x004

Page Table Lookup: VPN 0x403 → PFN 0x1234

Physical Address: (0x1234 * 4096) + 0x004 = 0x01234004
```

#### Multi-Level Page Tables

To save space, modern systems use hierarchical page tables:

```
64-bit Virtual Address (x86-64):
+-------+-------+-------+-------+--------+
| PML4  |  PDP  |  PD   |  PT   | Offset |
+-------+-------+-------+-------+--------+
  9 bits  9 bits  9 bits  9 bits  12 bits

Process:
1. Use PML4 index to find PDP table
2. Use PDP index to find PD table
3. Use PD index to find PT table
4. Use PT index to find physical frame
5. Add offset to get physical address
```

**Advantages:**
- Only allocate page tables for used memory
- Saves significant space compared to flat page table

#### Page Faults

A page fault occurs when accessing a virtual page not in physical memory.

**Types:**

**1. Minor (Soft) Page Fault**
- Page is in memory but not mapped in page table
- Fast to handle
- Example: First access to a newly allocated page

**2. Major (Hard) Page Fault**
- Page must be loaded from disk (swap)
- Very slow (milliseconds)
- Example: Accessing swapped-out memory

**3. Invalid Page Fault**
- Access to unmapped/protected memory
- Results in segmentation fault

**Page Fault Handling:**
```
1. CPU generates page fault exception
2. OS page fault handler runs
3. Check if address is valid
4. If valid:
   a. Find free physical frame
   b. Load page from disk (if needed)
   c. Update page table
   d. Restart instruction
5. If invalid:
   a. Terminate process (SIGSEGV)
```

**Example - Monitoring Page Faults (Linux):**
```bash
# Run command and show page fault statistics
/usr/bin/time -v ./myprogram

# Output includes:
# Major (requiring I/O) page faults: 123
# Minor (reclaiming a frame) page faults: 4567
```

#### Demand Paging

Pages are loaded into memory only when accessed (lazy loading).

**Benefits:**
- Programs can be larger than physical RAM
- Faster program startup (don't load everything)
- Better memory utilization

**Process:**
```c
int* big_array = malloc(1000000 * sizeof(int));
// Page tables created, but physical memory not allocated yet

big_array[0] = 42;  // Page fault! Allocate physical page
big_array[1000] = 100;  // Page fault! Allocate another page
```

#### Copy-on-Write (COW)

Optimization technique where multiple processes share the same physical pages until one writes to them.

**fork() Example:**
```c
int x = 42;  // Page containing x is marked COW

pid_t pid = fork();
// Child process shares parent's pages (read-only)

if (pid == 0) {
    // Child process
    x = 100;  // Write triggers COW:
              // 1. Page fault
              // 2. Copy page
              // 3. Update child's page table
              // 4. Mark both copies writable
}
```

**Benefits:**
- Fast fork() - no immediate copying
- Saves memory if pages not modified
- Common in modern Unix systems

#### Swap Space

When physical memory is full, OS can move pages to disk.

**Swapping Process:**
```
1. Select victim page (LRU, etc.)
2. Write page to swap space if dirty
3. Mark page table entry as swapped
4. Free physical frame
5. On access:
   a. Page fault
   b. Read from swap
   c. Allocate frame
   d. Update page table
```

**Performance Impact:**
```
Memory access: ~100 nanoseconds
Disk access: ~10 milliseconds
Ratio: 100,000x slower!
```

**Monitoring Swap (Linux):**
```bash
# Check swap usage
free -h

# Monitor swap activity
vmstat 1
```

#### Memory Protection

Virtual memory enables isolation and protection:

**Permission Bits:**
- **Read**: Can read from page
- **Write**: Can write to page
- **Execute**: Can execute code from page

**Example:**
```
Text segment: Read + Execute (no Write)
Data segment: Read + Write (no Execute)
Stack: Read + Write (no Execute on modern systems - NX bit)
```

**Protection Violation:**
```c
const char* str = "Hello";  // In read-only memory
str[0] = 'h';  // Segmentation fault! Write to read-only memory
```

#### Translation Lookaside Buffer (TLB)

Hardware cache for page table entries.

**Why Needed:**
- Page table lookups are expensive (multiple memory accesses)
- Most programs have high locality
- Cache recent translations

**Structure:**
```
Virtual Page Number → TLB Lookup
  ↓ Hit                ↓ Miss
Physical Frame    Page Table Walk
```

**TLB Miss Handling:**
- Hardware-managed (x86): CPU walks page table
- Software-managed (MIPS): OS exception handler

**Performance Impact:**
```c
// TLB-friendly: Sequential access
for (int i = 0; i < N; i++) {
    array[i] = i;  // High TLB hit rate
}

// TLB-unfriendly: Random access across many pages
for (int i = 0; i < N; i++) {
    int index = random() % N;
    array[index] = i;  // Many TLB misses
}
```

**Checking TLB Misses (Linux):**
```bash
perf stat -e dTLB-loads,dTLB-load-misses ./myprogram
```

### Memory Alignment

Memory alignment refers to arranging data in memory at addresses that are multiples of certain boundaries.

#### Why Alignment Matters

**1. Performance**
- Aligned accesses are faster on most architectures
- Unaligned accesses may require multiple memory operations
- Some CPUs (ARM) can crash on unaligned access

**2. Atomic Operations**
- Atomic operations often require aligned addresses
- Prevents word tearing

**3. Hardware Requirements**
- Some SIMD instructions require 16-byte or 32-byte alignment
- DMA operations may require specific alignment

#### Alignment Requirements by Type

```c
// Typical alignment requirements (x86-64)
char:    1-byte alignment  (address % 1 == 0)
short:   2-byte alignment  (address % 2 == 0)
int:     4-byte alignment  (address % 4 == 0)
long:    8-byte alignment  (address % 8 == 0)
float:   4-byte alignment  (address % 4 == 0)
double:  8-byte alignment  (address % 8 == 0)
pointer: 8-byte alignment  (address % 8 == 0) on 64-bit
```

#### Structure Padding

Compilers insert padding to maintain alignment:

**Example 1: Padding Between Fields**
```c
struct Example1 {
    char a;    // 1 byte
    // 3 bytes padding
    int b;     // 4 bytes (needs 4-byte alignment)
    char c;    // 1 byte
    // 3 bytes padding (for next struct in array)
};
// Total: 12 bytes (not 6!)

printf("Size: %zu\n", sizeof(struct Example1));  // 12
```

**Memory Layout:**
```
Offset: 0  1  2  3  4  5  6  7  8  9  10 11
       [a][  padding  ][   b      ][c][padding]
```

**Example 2: Reordering for Efficiency**
```c
// Inefficient layout
struct Bad {
    char a;    // 1 byte
    double b;  // 8 bytes (needs 8-byte alignment)
    char c;    // 1 byte
};
// Size: 24 bytes

// Efficient layout
struct Good {
    double b;  // 8 bytes
    char a;    // 1 byte
    char c;    // 1 byte
    // 6 bytes padding
};
// Size: 16 bytes (33% smaller!)
```

**Memory Layouts:**
```
Bad:
[a][      padding      ][        b        ][c][     padding     ]
1  +          7         +         8         + 1 +        7        = 24

Good:
[        b        ][a][c][    padding    ]
        8          + 1 + 1 +      6        = 16
```

#### Checking and Controlling Alignment

**Check Field Offsets:**
```c
#include <stddef.h>

struct Example {
    char a;
    int b;
    char c;
};

printf("Offset of a: %zu\n", offsetof(struct Example, a));  // 0
printf("Offset of b: %zu\n", offsetof(struct Example, b));  // 4
printf("Offset of c: %zu\n", offsetof(struct Example, c));  // 8
printf("Total size: %zu\n", sizeof(struct Example));        // 12
```

**Pack Structures (Remove Padding):**
```c
// GCC/Clang
struct __attribute__((packed)) Packed {
    char a;
    int b;
    char c;
};
// Size: 6 bytes (no padding)

// MSVC
#pragma pack(push, 1)
struct Packed {
    char a;
    int b;
    char c;
};
#pragma pack(pop)
```

**Warning:** Packed structures can cause:
- Slower access (unaligned reads)
- Crashes on some architectures (ARM)
- Inability to take aligned pointers

**Specify Alignment:**
```c
// Align to 16-byte boundary
struct alignas(16) Aligned {
    int x;
    int y;
};

// Or with GCC/Clang
struct __attribute__((aligned(16))) Aligned {
    int x;
    int y;
};
```

**C11 aligned_alloc:**
```c
// Allocate 64 bytes aligned to 32-byte boundary
void* ptr = aligned_alloc(32, 64);
if (ptr) {
    // Use ptr
    free(ptr);
}
```

**C++ alignas:**
```cpp
// Align variable to cache line (64 bytes)
alignas(64) int cache_aligned_var;

// Align structure
struct alignas(32) SimdData {
    float data[8];
};
```

#### Performance Impact

**Benchmark: Aligned vs Unaligned Access**
```c
#include <time.h>

// Aligned access
struct Aligned {
    int a;
    int b;
} __attribute__((aligned(8)));

// Unaligned access
struct __attribute__((packed)) Unaligned {
    char padding;
    int a;
    int b;
};

void benchmark_aligned() {
    struct Aligned data[1000000];
    clock_t start = clock();
    for (int i = 0; i < 1000000; i++) {
        data[i].a = i;
        data[i].b = i * 2;
    }
    clock_t end = clock();
    printf("Aligned: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}

void benchmark_unaligned() {
    struct Unaligned data[1000000];
    clock_t start = clock();
    for (int i = 0; i < 1000000; i++) {
        data[i].a = i;
        data[i].b = i * 2;
    }
    clock_t end = clock();
    printf("Unaligned: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}
```

**Typical Results:**
- x86-64: 10-50% slower for unaligned
- ARM: May crash or be 2-3x slower

#### SIMD Alignment

SIMD (Single Instruction Multiple Data) operations often require strict alignment:

```c
#include <immintrin.h>

// Must be 32-byte aligned for AVX
__attribute__((aligned(32))) float data[8];

// Load with alignment requirement
__m256 vec = _mm256_load_ps(data);  // Requires 32-byte alignment

// Load without alignment requirement (slower)
__m256 vec = _mm256_loadu_ps(data);  // Works with any alignment
```

### Fragmentation

Fragmentation occurs when memory is allocated and freed in a way that leaves unusable gaps.

#### Internal Fragmentation

Memory wasted within allocated blocks.

**Causes:**
- Alignment requirements
- Fixed-size allocation classes
- Rounding up allocations

**Example 1: Alignment**
```c
// Request 9 bytes, but allocator rounds to 16 for alignment
char* ptr = malloc(9);
// Actual allocation: 16 bytes
// Internal fragmentation: 7 bytes (43%!)
```

**Example 2: Size Classes**
```c
// Allocator has size classes: 8, 16, 32, 64, 128, 256...
char* small = malloc(17);
// Allocated from 32-byte class
// Internal fragmentation: 15 bytes
```

**Visualization:**
```
Requested: 9 bytes
Allocated: 16 bytes

[xxxxxxxxx-------]
 used     wasted (internal fragmentation)
```

**Measuring Internal Fragmentation:**
```
Internal Fragmentation = (Allocated - Requested) / Allocated

Example: (16 - 9) / 16 = 43.75%
```

#### External Fragmentation

Free memory exists but is scattered in small, non-contiguous blocks.

**Example Scenario:**
```c
// Initial state: 1000 bytes free
// [                1000 bytes free                    ]

char* a = malloc(100);
// [A:100][              900 bytes free               ]

char* b = malloc(100);
// [A:100][B:100][       800 bytes free              ]

char* c = malloc(100);
// [A:100][B:100][C:100][  700 bytes free           ]

free(b);
// [A:100][100 free][C:100][  700 bytes free        ]

// Now we have 800 bytes free total (100 + 700)
// But cannot allocate a 200-byte block!
char* d = malloc(200);  // Might fail or require compaction
```

**Visualization:**
```
Memory State:
[Allocated][Free:100][Allocated][Free:700]
            ↑                     ↑
            Small hole           Larger hole

Cannot satisfy 200-byte request despite having 800 bytes free!
```

**Measuring External Fragmentation:**
```
External Fragmentation = 1 - (Largest Free Block / Total Free Memory)

Example: 1 - (700 / 800) = 12.5%
```

#### Fragmentation Comparison

| Type | Where | Cause | Solution |
|------|-------|-------|----------|
| Internal | Within blocks | Alignment, size classes | Better size classes, packing |
| External | Between blocks | Allocation patterns | Compaction, better algorithms |

#### Reducing Internal Fragmentation

**1. Better Size Classes**
```c
// Poor size classes (power of 2)
// 8, 16, 32, 64, 128, 256...
// Requesting 65 bytes wastes 63 bytes (49%)

// Better size classes (more granular)
// 8, 12, 16, 24, 32, 48, 64, 96, 128...
// Requesting 65 bytes wastes 31 bytes (32%)
```

**2. Exact-Fit Allocations**
```c
// For known sizes, avoid overhead
struct Object {
    // Design to fit size class
    int data[6];  // 24 bytes - fits 32-byte class well
};
```

**3. Custom Allocators**
```c
// Pool allocator for fixed-size objects
// Zero internal fragmentation
struct Pool {
    void* free_list;
    size_t object_size;
};
```

#### Reducing External Fragmentation

**1. Buddy Allocation**

Splits memory into power-of-2 blocks that can be merged.

```
Initial: [              128 bytes              ]

Request 16 bytes:
Split:   [      64      ][      64      ]
Split:   [  32  ][  32  ][      64      ]
Split:   [16][16][  32  ][      64      ]
Allocate:[A ][ F][  F   ][      F       ]

Request 32 bytes:
Allocate:[A ][ F][  B   ][      F       ]

Free A:
State:   [ F][ F][  B   ][      F       ]
Merge:   [  F   ][  B   ][      F       ]

Free B:
State:   [  F   ][  F   ][      F       ]
Merge:   [              F              ]
```

**2. Best-Fit Allocation**
```c
// Find smallest block that fits request
// Minimizes wasted space
void* best_fit(size_t size, struct FreeList* list) {
    struct Block* best = NULL;
    size_t best_size = SIZE_MAX;

    for (struct Block* b = list->head; b; b = b->next) {
        if (b->size >= size && b->size < best_size) {
            best = b;
            best_size = b->size;
        }
    }
    return best;
}
```

**3. First-Fit Allocation**
```c
// Use first block that fits
// Faster than best-fit
void* first_fit(size_t size, struct FreeList* list) {
    for (struct Block* b = list->head; b; b = b->next) {
        if (b->size >= size) {
            return b;
        }
    }
    return NULL;
}
```

**4. Memory Compaction**

Move allocated blocks together to consolidate free space.

```
Before:
[A][Free][B][Free][C][Free]

After compaction:
[A][B][C][        Free       ]
```

**Challenge:** Must update all pointers to moved objects!

**Solutions:**
- Handles/indirect pointers (Java, Go)
- Moving GC with pointer tracking
- Generally not possible in C/C++

**5. Segregated Free Lists**

Maintain separate free lists for different size classes.

```c
struct Allocator {
    struct FreeList* lists[NUM_SIZE_CLASSES];
};

// Size classes: 16, 32, 64, 128, 256, 512, 1024, 2048...
void* allocate(struct Allocator* alloc, size_t size) {
    int class = size_class(size);
    if (alloc->lists[class]->head) {
        return allocate_from_list(alloc->lists[class]);
    }
    // Fall back to larger class or request from OS
}
```

**Advantages:**
- Fast allocation (no search)
- Reduced fragmentation within classes
- Better cache locality

#### Real-World Example: jemalloc

jemalloc uses multiple techniques:

```
Size Class Ranges:
- Small: 8, 16, 32, 48, 64, 80, 96, 112, 128... (up to 14 KB)
  → Segregated free lists, thread-local caching

- Large: 16 KB, 32 KB, 48 KB... (up to 4 MB)
  → Best-fit allocation

- Huge: > 4 MB
  → Direct mmap() calls

Arenas:
- Multiple per thread to reduce contention
- Each arena has own metadata

Result:
- Low fragmentation (typically < 10%)
- Good performance
```

#### Monitoring Fragmentation

**Linux - /proc/meminfo:**
```bash
$ cat /proc/meminfo | grep Frag
# Shows fragmentation index (0 = no fragmentation, 1 = max)
```

**Malloc Statistics (GNU libc):**
```c
#include <malloc.h>

struct mallinfo info = mallinfo();
printf("Total allocated: %d\n", info.uordblks);
printf("Total free: %d\n", info.fordblks);
printf("Fragmentation: %.2f%%\n",
       100.0 * info.fordblks / (info.uordblks + info.fordblks));
```

**Custom Tracking:**
```c
size_t total_requested = 0;
size_t total_allocated = 0;

void* my_malloc(size_t size) {
    size_t actual = round_up(size);
    total_requested += size;
    total_allocated += actual;

    double internal_frag = 100.0 * (total_allocated - total_requested)
                                 / total_allocated;
    printf("Internal fragmentation: %.2f%%\n", internal_frag);

    return malloc(actual);
}
```

---

## Allocation Strategies

### Static Allocation

Memory allocated at compile time and exists for the program's entire lifetime.

#### Characteristics

- **Lifetime**: Program start to program end
- **Location**: Data or BSS segment
- **Size**: Fixed at compile time
- **Speed**: No runtime overhead
- **Thread-safety**: Potential issues with shared mutable state

#### Types of Static Allocation

**1. Global Variables**
```c
// Initialized global (data segment)
int global_counter = 0;

// Uninitialized global (BSS segment)
int global_array[1000];

void function() {
    global_counter++;  // Direct access, very fast
}
```

**2. Static Local Variables**
```c
void function() {
    // Initialized once, persists across calls
    static int call_count = 0;
    call_count++;

    printf("Called %d times\n", call_count);
}

// First call: "Called 1 times"
// Second call: "Called 2 times"
```

**3. String Literals**
```c
// String literal in read-only data segment
const char* message = "Hello, World!";

// Array initialized with string literal
char buffer[] = "Hello";  // Mutable copy on stack
```

**4. Static Arrays**
```c
// Large lookup table
static const int fibonacci[20] = {
    0, 1, 1, 2, 3, 5, 8, 13, 21, 34,
    55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181
};

int get_fibonacci(int n) {
    return fibonacci[n];  // O(1) lookup
}
```

#### Advantages

**1. Performance**
```c
// Static: no allocation overhead
static int cache[1000];

// vs Dynamic: allocation overhead every time
int* cache = malloc(1000 * sizeof(int));
```

**2. Simplicity**
```c
// No need to manage lifetime
static const char* error_messages[] = {
    "Success",
    "File not found",
    "Permission denied",
    "Out of memory"
};
```

**3. Guaranteed Initialization**
```c
// BSS guarantees zero-initialization
static int counters[100];  // All zeros
static char buffer[1024];  // All zeros
```

#### Disadvantages

**1. Memory Usage**
```c
// Always allocated, even if never used
static char huge_buffer[1000000];  // 1 MB always consumed

void rarely_called_function() {
    // This buffer exists even if function never called
}
```

**2. No Dynamic Sizing**
```c
// Must define maximum size at compile time
#define MAX_USERS 1000
static struct User users[MAX_USERS];

// Cannot grow beyond MAX_USERS
```

**3. Thread-Safety Issues**
```c
// Global state is shared across threads
static int counter = 0;

void increment() {
    counter++;  // Race condition!
}

// Solution: use thread-local storage
_Thread_local int counter = 0;  // C11
// or
thread_local int counter = 0;    // C++11
```

#### Use Cases

**1. Lookup Tables**
```c
static const unsigned char reverse_bits[256] = {
    0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0,
    // ... precomputed values
};

unsigned char reverse(unsigned char b) {
    return reverse_bits[b];
}
```

**2. Singleton Pattern**
```c
struct Logger* get_logger() {
    static struct Logger instance = {0};
    static int initialized = 0;

    if (!initialized) {
        logger_init(&instance);
        initialized = 1;
    }

    return &instance;
}
```

**3. String Constants**
```c
const char* get_version() {
    return "1.0.0";  // String literal (static)
}
```

**4. State Machines**
```c
enum State { START, RUNNING, STOPPED };

void state_machine() {
    static enum State current = START;

    switch (current) {
        case START:
            // ...
            current = RUNNING;
            break;
        case RUNNING:
            // ...
            break;
        case STOPPED:
            // ...
            break;
    }
}
```

### Stack Allocation

Memory automatically allocated on the call stack when entering a function and freed when exiting.

#### Characteristics

- **Lifetime**: Function scope
- **Location**: Stack segment
- **Size**: Must be known at compile time (typically)
- **Speed**: Extremely fast (just move stack pointer)
- **Cleanup**: Automatic

#### Basic Stack Allocation

```c
void function() {
    int x = 42;              // 4 bytes on stack
    char buffer[100];        // 100 bytes on stack
    double values[10];       // 80 bytes on stack

    struct Point {
        int x, y;
    } p = {1, 2};           // 8 bytes on stack

}  // All automatically freed here
```

#### Variable Length Arrays (VLA) - C99

```c
void process(int n) {
    // Stack-allocated array with runtime size
    int array[n];  // VLA

    for (int i = 0; i < n; i++) {
        array[i] = i * i;
    }

}  // array automatically freed

// Warning: Dangerous for large n (stack overflow)
process(1000000);  // May crash!
```

**VLA Limitations:**
- Not supported in C++ (except as compiler extension)
- Dangerous for large sizes
- Size must fit in stack (typically 1-8 MB)
- No way to check if allocation succeeded

#### alloca() - Dynamic Stack Allocation

```c
#include <alloca.h>

void function(size_t n) {
    // Allocate n bytes on stack
    char* buffer = alloca(n);

    // Use buffer...
    memset(buffer, 0, n);

}  // buffer automatically freed

// Warning: Same dangers as VLA
```

**Why alloca() is dangerous:**
- No error checking (can't detect failure)
- Stack overflow crashes program
- Not portable (POSIX, not C standard)
- Can't be used in loops safely

```c
// DANGEROUS: unbounded stack growth
for (int i = 0; i < n; i++) {
    char* buf = alloca(1000);  // Stack grows each iteration!
    // Memory not freed until function returns!
}
```

#### Advantages of Stack Allocation

**1. Speed**
```
Stack allocation: ~1 nanosecond
Heap allocation: ~100 nanoseconds
Ratio: 100x faster!
```

**2. Automatic Cleanup**
```c
void function() {
    char buffer[1024];

    if (error_condition) {
        return;  // buffer automatically cleaned up
    }

    // Use buffer...

}  // buffer automatically cleaned up
```

**3. Cache-Friendly**
```c
// Stack-allocated data has good locality
void process() {
    int a = 1;
    int b = 2;
    int c = 3;
    // a, b, c likely in same cache line
}
```

**4. No Fragmentation**
```c
// Stack pointer just moves up/down
// No fragmentation issues
```

#### Disadvantages of Stack Allocation

**1. Limited Size**
```bash
# Check stack size limit (Linux)
$ ulimit -s
8192  # 8 MB default

# Set larger stack size
$ ulimit -s 16384  # 16 MB
```

```c
// Stack overflow example
void recursive(int n) {
    char buffer[1024];  // 1 KB per call
    recursive(n + 1);   // Eventually crashes
}
```

**2. Lifetime Limitations**
```c
char* create_string() {
    char buffer[100] = "Hello";
    return buffer;  // BUG! Returning pointer to stack memory
}

// Usage:
char* str = create_string();
printf("%s\n", str);  // Undefined behavior!
```

**3. Size Must Be Known**
```c
void function(int n) {
    // Can't do this (without VLA):
    // int array[n];  // Not allowed in C++

    // Must use heap:
    int* array = new int[n];
    // ...
    delete[] array;
}
```

#### Best Practices

**1. Prefer Stack for Small, Short-Lived Data**
```c
// Good: small buffer, short lifetime
void process_line(const char* line) {
    char buffer[256];
    strncpy(buffer, line, 255);
    buffer[255] = '\0';
    // Process buffer...
}
```

**2. Use Heap for Large Data**
```c
// Bad: large stack allocation
void bad() {
    char buffer[1000000];  // 1 MB - risky!
}

// Good: use heap
void good() {
    char* buffer = malloc(1000000);
    if (!buffer) {
        // Handle error
        return;
    }
    // Use buffer...
    free(buffer);
}
```

**3. Avoid Returning Stack Addresses**
```c
// Bad
char* get_message() {
    char buffer[100] = "Hello";
    return buffer;  // Dangling pointer!
}

// Good: return string literal (static)
const char* get_message() {
    return "Hello";
}

// Good: use heap
char* get_message() {
    char* buffer = malloc(100);
    strcpy(buffer, "Hello");
    return buffer;  // Caller must free()
}

// Good: use output parameter
void get_message(char* buffer, size_t size) {
    strncpy(buffer, "Hello", size - 1);
    buffer[size - 1] = '\0';
}
```

**4. Check Stack Usage**
```c
#include <sys/resource.h>

void check_stack() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("Max stack size: %ld KB\n", usage.ru_maxrss);
}
```

### Heap Allocation

Dynamic memory allocation from the heap at runtime.

#### Basic Heap Allocation

**C:**
```c
#include <stdlib.h>

// Allocate
int* ptr = malloc(sizeof(int) * 10);
if (!ptr) {
    // Handle allocation failure
    return;
}

// Use
ptr[0] = 42;

// Free
free(ptr);
ptr = NULL;  // Avoid dangling pointer
```

**C++:**
```cpp
// Allocate single object
int* ptr = new int(42);
delete ptr;

// Allocate array
int* arr = new int[10];
delete[] arr;

// Modern C++: use smart pointers instead
std::unique_ptr<int> ptr = std::make_unique<int>(42);
std::unique_ptr<int[]> arr = std::make_unique<int[]>(10);
// Automatic cleanup
```

#### Memory Allocation Functions (C)

**malloc():**
```c
void* malloc(size_t size);

// Allocates uninitialized memory
int* ptr = malloc(sizeof(int) * 100);
// Memory contains garbage values
```

**calloc():**
```c
void* calloc(size_t count, size_t size);

// Allocates zero-initialized memory
int* ptr = calloc(100, sizeof(int));
// All elements are 0
```

**Performance:**
```c
// malloc: faster (no initialization)
int* a = malloc(1000000 * sizeof(int));

// calloc: slower (zeros memory)
int* b = calloc(1000000, sizeof(int));
```

**realloc():**
```c
void* realloc(void* ptr, size_t new_size);

// Resize allocation
int* ptr = malloc(sizeof(int) * 10);
// ...
ptr = realloc(ptr, sizeof(int) * 20);  // Grow to 20 elements

if (!ptr) {
    // realloc failed, original pointer still valid
    // (unless ptr was NULL)
}
```

**realloc() Behavior:**
```c
// 1. new_size > old_size: may move and copy data
// 2. new_size < old_size: may shrink in place
// 3. new_size == 0: equivalent to free()
// 4. ptr == NULL: equivalent to malloc()

// Example: growing array
int* arr = NULL;
size_t capacity = 0;

for (int i = 0; i < 100; i++) {
    if (i >= capacity) {
        capacity = capacity ? capacity * 2 : 1;
        int* new_arr = realloc(arr, capacity * sizeof(int));
        if (!new_arr) {
            free(arr);
            return;  // Handle error
        }
        arr = new_arr;
    }
    arr[i] = i;
}
```

**free():**
```c
void free(void* ptr);

// Free allocated memory
int* ptr = malloc(sizeof(int));
free(ptr);

// Safe to free NULL
free(NULL);  // No-op

// Double-free is undefined behavior
free(ptr);
free(ptr);  // BUG!

// Best practice: NULL after free
free(ptr);
ptr = NULL;
```

#### Alignment and Allocation

**aligned_alloc() - C11:**
```c
void* aligned_alloc(size_t alignment, size_t size);

// Allocate with specific alignment
// size must be multiple of alignment
void* ptr = aligned_alloc(64, 128);  // 64-byte aligned, 128 bytes
free(ptr);
```

**posix_memalign():**
```c
int posix_memalign(void** ptr, size_t alignment, size_t size);

void* ptr;
if (posix_memalign(&ptr, 64, 128) != 0) {
    // Handle error
}
free(ptr);
```

#### Allocation Patterns

**Pattern 1: Fixed-Size Allocations**
```c
struct Node {
    int data;
    struct Node* next;
};

struct Node* create_node(int data) {
    struct Node* node = malloc(sizeof(struct Node));
    if (node) {
        node->data = data;
        node->next = NULL;
    }
    return node;
}
```

**Pattern 2: Variable-Size Allocations**
```c
struct String {
    size_t length;
    char* data;
};

struct String* create_string(const char* str) {
    struct String* s = malloc(sizeof(struct String));
    if (!s) return NULL;

    s->length = strlen(str);
    s->data = malloc(s->length + 1);
    if (!s->data) {
        free(s);
        return NULL;
    }

    strcpy(s->data, str);
    return s;
}

void free_string(struct String* s) {
    if (s) {
        free(s->data);
        free(s);
    }
}
```

**Pattern 3: Flexible Array Members (C99)**
```c
struct Buffer {
    size_t size;
    char data[];  // Flexible array member
};

struct Buffer* create_buffer(size_t size) {
    // Allocate structure + array in one block
    struct Buffer* buf = malloc(sizeof(struct Buffer) + size);
    if (buf) {
        buf->size = size;
    }
    return buf;
}

void free_buffer(struct Buffer* buf) {
    free(buf);  // Single free for both struct and array
}
```

**Pattern 4: Growing Arrays**
```c
struct DynamicArray {
    int* data;
    size_t size;
    size_t capacity;
};

void push_back(struct DynamicArray* arr, int value) {
    if (arr->size >= arr->capacity) {
        size_t new_capacity = arr->capacity ? arr->capacity * 2 : 1;
        int* new_data = realloc(arr->data, new_capacity * sizeof(int));
        if (!new_data) {
            // Handle error
            return;
        }
        arr->data = new_data;
        arr->capacity = new_capacity;
    }
    arr->data[arr->size++] = value;
}
```

#### Allocation Performance

**Size Class Optimization:**

Modern allocators use size classes to reduce overhead:

```
jemalloc size classes (examples):
Small: 8, 16, 24, 32, 48, 64, 80, 96, 112, 128...
Large: 4K, 8K, 12K, 16K, 20K, 24K...
Huge: > 4MB (direct mmap)
```

**Implications:**
```c
// Request 17 bytes → get 24 bytes (from 24-byte class)
char* small = malloc(17);  // 7 bytes wasted

// Request 4097 bytes → get 8K (from 8K class)
char* medium = malloc(4097);  // ~4K wasted!

// Request 10 MB → direct mmap, exact size
char* large = malloc(10 * 1024 * 1024);
```

**Allocation Overhead:**
```c
// Each allocation has metadata overhead
struct BlockHeader {
    size_t size;
    int flags;
    // Maybe more fields
};  // Typically 8-16 bytes

// So allocating 1 byte actually uses ~16 bytes!
char* tiny = malloc(1);  // 1 byte + 16 byte overhead = 17 bytes
```

**Reducing Overhead:**
```c
// Bad: many small allocations
for (int i = 0; i < 1000; i++) {
    int* p = malloc(sizeof(int));  // 1000 allocations
}

// Good: single large allocation
int* array = malloc(sizeof(int) * 1000);  // 1 allocation
```

### Memory Pools

Pre-allocated memory blocks for fixed-size objects, providing fast, predictable allocation.

#### Basic Concept

```
Memory Pool:
[Free][Free][Free][Used][Free][Used][Used][Free]
  ↓
Free List: → [0] → [1] → [2] → [4] → [7] → NULL
```

#### Simple Pool Implementation

```c
#define POOL_SIZE 1000
#define OBJECT_SIZE sizeof(struct Object)

struct Pool {
    void* memory;
    void* free_list;
    size_t object_size;
    size_t capacity;
};

struct FreeNode {
    struct FreeNode* next;
};

// Initialize pool
struct Pool* pool_create(size_t object_size, size_t capacity) {
    struct Pool* pool = malloc(sizeof(struct Pool));
    if (!pool) return NULL;

    pool->memory = malloc(object_size * capacity);
    if (!pool->memory) {
        free(pool);
        return NULL;
    }

    pool->object_size = object_size;
    pool->capacity = capacity;

    // Build free list
    pool->free_list = pool->memory;
    char* ptr = pool->memory;
    for (size_t i = 0; i < capacity - 1; i++) {
        struct FreeNode* node = (struct FreeNode*)ptr;
        node->next = (struct FreeNode*)(ptr + object_size);
        ptr += object_size;
    }
    ((struct FreeNode*)ptr)->next = NULL;

    return pool;
}

// Allocate from pool
void* pool_alloc(struct Pool* pool) {
    if (!pool->free_list) {
        return NULL;  // Pool exhausted
    }

    void* ptr = pool->free_list;
    pool->free_list = ((struct FreeNode*)ptr)->next;
    return ptr;
}

// Free back to pool
void pool_free(struct Pool* pool, void* ptr) {
    struct FreeNode* node = (struct FreeNode*)ptr;
    node->next = pool->free_list;
    pool->free_list = node;
}

// Destroy pool
void pool_destroy(struct Pool* pool) {
    free(pool->memory);
    free(pool);
}
```

#### Usage Example

```c
struct Node {
    int data;
    struct Node* left;
    struct Node* right;
};

int main() {
    // Create pool for 1000 nodes
    struct Pool* node_pool = pool_create(sizeof(struct Node), 1000);

    // Allocate nodes from pool (very fast!)
    struct Node* n1 = pool_alloc(node_pool);
    struct Node* n2 = pool_alloc(node_pool);
    struct Node* n3 = pool_alloc(node_pool);

    n1->data = 1;
    n1->left = n2;
    n1->right = n3;

    // Free nodes back to pool
    pool_free(node_pool, n1);
    pool_free(node_pool, n2);
    pool_free(node_pool, n3);

    // Destroy pool
    pool_destroy(node_pool);

    return 0;
}
```

#### Performance Benefits

```c
// Benchmark: malloc vs pool allocation

// Using malloc
clock_t start = clock();
for (int i = 0; i < 1000000; i++) {
    void* p = malloc(32);
    free(p);
}
clock_t malloc_time = clock() - start;

// Using pool
struct Pool* pool = pool_create(32, 1000000);
start = clock();
void* pointers[1000000];
for (int i = 0; i < 1000000; i++) {
    pointers[i] = pool_alloc(pool);
}
for (int i = 0; i < 1000000; i++) {
    pool_free(pool, pointers[i]);
}
clock_t pool_time = clock() - start;

printf("malloc: %f seconds\n", (double)malloc_time / CLOCKS_PER_SEC);
printf("pool: %f seconds\n", (double)pool_time / CLOCKS_PER_SEC);
printf("Speedup: %.2fx\n", (double)malloc_time / pool_time);

// Typical result: 10-50x faster!
```

#### Advantages

1. **Speed**: O(1) allocation/deallocation
2. **No fragmentation**: All objects same size
3. **Predictable performance**: No syscalls
4. **Cache-friendly**: Objects allocated together
5. **No individual overhead**: Metadata only for pool, not each object

#### Disadvantages

1. **Fixed object size**: Can't allocate different sizes
2. **Wasted memory**: Unused pool capacity
3. **Manual management**: Must return objects to pool
4. **Pool exhaustion**: Can run out of objects

#### Use Cases

- Game engines (entities, particles)
- Network servers (connection objects)
- Database systems (query nodes)
- Any system with many fixed-size allocations

### Arena Allocators

Region-based memory management where allocations are freed all at once.

#### Basic Concept

```
Arena:
[Allocation 1][Allocation 2][Allocation 3][  Free Space  ]
                                           ↑
                                         Current position

Free all at once:
[                    All Free                            ]
```

#### Simple Arena Implementation

```c
struct Arena {
    char* buffer;
    size_t size;
    size_t used;
};

// Create arena
struct Arena* arena_create(size_t size) {
    struct Arena* arena = malloc(sizeof(struct Arena));
    if (!arena) return NULL;

    arena->buffer = malloc(size);
    if (!arena->buffer) {
        free(arena);
        return NULL;
    }

    arena->size = size;
    arena->used = 0;

    return arena;
}

// Allocate from arena
void* arena_alloc(struct Arena* arena, size_t size) {
    // Align to 8-byte boundary
    size_t aligned_size = (size + 7) & ~7;

    if (arena->used + aligned_size > arena->size) {
        return NULL;  // Arena full
    }

    void* ptr = arena->buffer + arena->used;
    arena->used += aligned_size;

    return ptr;
}

// Reset arena (free all allocations)
void arena_reset(struct Arena* arena) {
    arena->used = 0;
}

// Destroy arena
void arena_destroy(struct Arena* arena) {
    free(arena->buffer);
    free(arena);
}
```

#### Usage Example

```c
void process_request(struct Request* request) {
    // Create arena for this request
    struct Arena* arena = arena_create(1024 * 1024);  // 1 MB

    // Allocate temporary data from arena
    char* buffer = arena_alloc(arena, 4096);
    struct Parser* parser = arena_alloc(arena, sizeof(struct Parser));
    struct AST* ast = arena_alloc(arena, sizeof(struct AST));

    // Process request using allocated data
    parse_request(parser, request, buffer);
    build_ast(ast, parser);
    execute_ast(ast);

    // Free everything at once!
    arena_destroy(arena);
    // No need to free buffer, parser, ast individually
}
```

#### Advanced Arena with Growing

```c
struct ArenaBlock {
    char* buffer;
    size_t size;
    size_t used;
    struct ArenaBlock* next;
};

struct GrowingArena {
    struct ArenaBlock* current;
    size_t default_block_size;
};

struct GrowingArena* growing_arena_create(size_t default_size) {
    struct GrowingArena* arena = malloc(sizeof(struct GrowingArena));
    if (!arena) return NULL;

    arena->default_block_size = default_size;
    arena->current = calloc(1, sizeof(struct ArenaBlock));
    if (!arena->current) {
        free(arena);
        return NULL;
    }

    arena->current->buffer = malloc(default_size);
    if (!arena->current->buffer) {
        free(arena->current);
        free(arena);
        return NULL;
    }

    arena->current->size = default_size;
    arena->current->used = 0;
    arena->current->next = NULL;

    return arena;
}

void* growing_arena_alloc(struct GrowingArena* arena, size_t size) {
    size_t aligned_size = (size + 7) & ~7;

    // Try current block
    if (arena->current->used + aligned_size <= arena->current->size) {
        void* ptr = arena->current->buffer + arena->current->used;
        arena->current->used += aligned_size;
        return ptr;
    }

    // Need new block
    size_t block_size = arena->default_block_size;
    if (aligned_size > block_size) {
        block_size = aligned_size;
    }

    struct ArenaBlock* new_block = calloc(1, sizeof(struct ArenaBlock));
    if (!new_block) return NULL;

    new_block->buffer = malloc(block_size);
    if (!new_block->buffer) {
        free(new_block);
        return NULL;
    }

    new_block->size = block_size;
    new_block->used = aligned_size;
    new_block->next = arena->current;
    arena->current = new_block;

    return new_block->buffer;
}

void growing_arena_destroy(struct GrowingArena* arena) {
    struct ArenaBlock* block = arena->current;
    while (block) {
        struct ArenaBlock* next = block->next;
        free(block->buffer);
        free(block);
        block = next;
    }
    free(arena);
}
```

#### Performance Characteristics

```c
// Benchmark: malloc vs arena

// Using malloc (must track and free each allocation)
clock_t start = clock();
char* pointers[10000];
for (int i = 0; i < 10000; i++) {
    pointers[i] = malloc(100);
}
for (int i = 0; i < 10000; i++) {
    free(pointers[i]);
}
clock_t malloc_time = clock() - start;

// Using arena
start = clock();
struct Arena* arena = arena_create(10000 * 100);
for (int i = 0; i < 10000; i++) {
    arena_alloc(arena, 100);
}
arena_destroy(arena);  // Free all at once!
clock_t arena_time = clock() - start;

printf("malloc: %f seconds\n", (double)malloc_time / CLOCKS_PER_SEC);
printf("arena: %f seconds\n", (double)arena_time / CLOCKS_PER_SEC);
printf("Speedup: %.2fx\n", (double)malloc_time / arena_time);

// Typical result: 5-20x faster!
```

#### Advantages

1. **Very fast allocation**: Just bump pointer
2. **Very fast deallocation**: Free all at once
3. **No fragmentation**: Linear allocation
4. **Simple implementation**: Minimal code
5. **Cache-friendly**: Sequential allocations
6. **No individual overhead**: No per-allocation metadata

#### Disadvantages

1. **Can't free individual objects**: All or nothing
2. **Memory usage**: Can't reclaim until reset/destroy
3. **Requires discipline**: Must reset/destroy appropriately
4. **Not general-purpose**: Specific use patterns

#### Use Cases

1. **Per-request processing**
```c
void handle_http_request(struct Request* req) {
    struct Arena* arena = arena_create(1024 * 1024);

    // Parse headers (allocates from arena)
    struct Headers* headers = parse_headers(arena, req);

    // Parse body (allocates from arena)
    struct Body* body = parse_body(arena, req);

    // Generate response (allocates from arena)
    struct Response* response = generate_response(arena, headers, body);

    // Send response
    send_response(response);

    // Free everything!
    arena_destroy(arena);
}
```

2. **Compiler phases**
```c
void compile(const char* source) {
    // Lexing phase
    struct Arena* lex_arena = arena_create(1024 * 1024);
    struct Token* tokens = lex(lex_arena, source);

    // Parsing phase
    struct Arena* parse_arena = arena_create(1024 * 1024);
    struct AST* ast = parse(parse_arena, tokens);
    arena_destroy(lex_arena);  // Don't need tokens anymore

    // Code generation
    struct Arena* codegen_arena = arena_create(1024 * 1024);
    struct Code* code = codegen(codegen_arena, ast);
    arena_destroy(parse_arena);  // Don't need AST anymore

    // Emit code
    emit(code);
    arena_destroy(codegen_arena);
}
```

3. **Game frames**
```c
void game_loop() {
    struct Arena* frame_arena = arena_create(10 * 1024 * 1024);

    while (running) {
        // Allocate temporary data for this frame
        struct RenderList* render_list = arena_alloc(frame_arena, sizeof(*render_list));
        struct Input* input = arena_alloc(frame_arena, sizeof(*input));

        // Process frame
        process_input(input);
        update_game_state(input);
        build_render_list(render_list);
        render(render_list);

        // Reset arena for next frame
        arena_reset(frame_arena);
    }

    arena_destroy(frame_arena);
}
```

#### Temporary Allocations Pattern

```c
struct ArenaSave {
    size_t used;
};

// Save arena state
struct ArenaSave arena_save(struct Arena* arena) {
    return (struct ArenaSave){ .used = arena->used };
}

// Restore arena state (free allocations since save)
void arena_restore(struct Arena* arena, struct ArenaSave save) {
    arena->used = save.used;
}

// Usage:
void function() {
    struct ArenaSave save = arena_save(arena);

    // Make temporary allocations
    char* temp1 = arena_alloc(arena, 100);
    char* temp2 = arena_alloc(arena, 200);

    // Use temporaries...

    // Restore (free temp1 and temp2)
    arena_restore(arena, save);
}
```

---

## Garbage Collection

Automatic memory management where the runtime system reclaims unused memory.

### Reference Counting

Track how many references point to each object; free when count reaches zero.

#### Basic Concept

```
Object: [data][ref_count=0]
         ↑
         |(create)
Object: [data][ref_count=1]
         ↑     ↑
         |     |(add reference)
Object: [data][ref_count=2]
         ↑
         |(remove reference)
Object: [data][ref_count=1]
         ↑
         |(remove reference)
Object: [data][ref_count=0] → FREE!
```

#### Simple Reference Counting Implementation

```c
struct RefCounted {
    void* data;
    size_t ref_count;
    void (*destructor)(void*);
};

// Create object with ref_count = 1
struct RefCounted* rc_create(void* data, void (*destructor)(void*)) {
    struct RefCounted* obj = malloc(sizeof(struct RefCounted));
    if (!obj) return NULL;

    obj->data = data;
    obj->ref_count = 1;
    obj->destructor = destructor;

    return obj;
}

// Increment reference count
void rc_retain(struct RefCounted* obj) {
    if (obj) {
        obj->ref_count++;
    }
}

// Decrement reference count; free if reaches 0
void rc_release(struct RefCounted* obj) {
    if (!obj) return;

    obj->ref_count--;

    if (obj->ref_count == 0) {
        if (obj->destructor) {
            obj->destructor(obj->data);
        }
        free(obj);
    }
}

// Usage example
void example() {
    // Create object (ref_count = 1)
    struct RefCounted* obj = rc_create(strdup("Hello"), free);

    // Share object (ref_count = 2)
    struct RefCounted* obj2 = obj;
    rc_retain(obj2);

    // Release first reference (ref_count = 1)
    rc_release(obj);

    // Release second reference (ref_count = 0, freed!)
    rc_release(obj2);
}
```

#### Python's Reference Counting

Python uses reference counting as its primary GC mechanism:

```python
import sys

# Create object (ref_count = 1)
a = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (1 + 1 for the argument to getrefcount)

# Add reference (ref_count = 2)
b = a
print(sys.getrefcount(a))  # 3

# Remove reference (ref_count = 1)
del b
print(sys.getrefcount(a))  # 2

# Remove last reference (ref_count = 0, freed!)
del a
```

**Python's Implementation (CPython):**
```c
// From Python's object.h (simplified)
typedef struct _object {
    Py_ssize_t ob_refcnt;  // Reference count
    struct _typeobject *ob_type;
} PyObject;

// Increment reference
#define Py_INCREF(op) ((void)(((PyObject*)(op))->ob_refcnt++))

// Decrement reference; free if 0
#define Py_DECREF(op) \
    do { \
        PyObject *_py_decref_tmp = (PyObject *)(op); \
        if (--(_py_decref_tmp)->ob_refcnt == 0) \
            _Py_Dealloc(_py_decref_tmp); \
    } while (0)
```

#### Swift's Automatic Reference Counting (ARC)

Swift automatically inserts retain/release calls at compile time:

```swift
class Person {
    var name: String
    init(name: String) { self.name = name }
    deinit { print("\(name) is being deinitialized") }
}

do {
    let person1 = Person(name: "John")  // ref_count = 1
    let person2 = person1                // ref_count = 2
}  // Scope ends: ref_count = 0, deinit called
```

**Compiler transforms to (conceptually):**
```swift
do {
    let person1 = Person(name: "John")
    swift_retain(person1)  // Inserted by compiler

    let person2 = person1
    swift_retain(person2)  // Inserted by compiler

    swift_release(person2)  // Inserted by compiler
    swift_release(person1)  // Inserted by compiler
}
```

#### Advantages of Reference Counting

1. **Deterministic**: Objects freed immediately when unreferenced
2. **No pause times**: No stop-the-world collection
3. **Simple**: Easy to understand and implement
4. **Incremental**: Work distributed over time

#### Disadvantages of Reference Counting

**1. Overhead**
```c
// Every pointer assignment requires ref count update
obj->field = new_value;  // Becomes:
rc_release(obj->field);
obj->field = new_value;
rc_retain(obj->field);
```

**2. Performance**
```c
// Cache pressure from updating ref counts
// False sharing in multithreaded code
```

**3. Cannot Handle Cycles**

```c
struct Node {
    struct RefCounted* parent;
    struct RefCounted* child;
};

// Create cycle
struct RefCounted* node1 = rc_create(...);
struct RefCounted* node2 = rc_create(...);

((struct Node*)node1->data)->child = node2;
rc_retain(node2);  // node2 ref_count = 2

((struct Node*)node2->data)->parent = node1;
rc_retain(node1);  // node1 ref_count = 2

// Release external references
rc_release(node1);  // node1 ref_count = 1 (still referenced by node2)
rc_release(node2);  // node2 ref_count = 1 (still referenced by node1)

// MEMORY LEAK! Both objects keep each other alive
```

**Visualization:**
```
node1 [ref_count=1] → node2 [ref_count=1]
  ↑                     ↓
  └─────────────────────┘

Cannot be freed because ref_count > 0 for both!
```

#### Solving Cycle Problem

**Solution 1: Weak References**

```swift
class Node {
    var value: Int
    var children: [Node] = []
    weak var parent: Node?  // Weak reference doesn't increment ref_count
}

let parent = Node(value: 1)     // ref_count = 1
let child = Node(value: 2)      // ref_count = 1
child.parent = parent           // parent ref_count still 1 (weak!)
parent.children.append(child)   // child ref_count = 2

// When parent goes out of scope:
// parent ref_count = 0, freed
// child.parent automatically becomes nil
// child ref_count = 1
```

**Solution 2: Cycle Detection (Python)**

Python combines reference counting with cycle detection:

```python
# Create cycle
class Node:
    pass

a = Node()
b = Node()
a.ref = b  # b ref_count = 2
b.ref = a  # a ref_count = 2

del a  # a ref_count = 1
del b  # b ref_count = 1

# Cycle detector (runs periodically) finds and breaks cycle
```

**Python's Cycle Detector:**
```c
// Simplified algorithm
1. Find all objects with ref_count > 0
2. Subtract internal references (between tracked objects)
3. Objects with effective ref_count = 0 are in cycles
4. Free them
```

#### Reference Counting in Practice

**Objective-C/Swift:**
- ARC automatically manages ref counts
- Weak references for breaking cycles
- `@autoreleasepool` for optimization

**C++ `std::shared_ptr`:**
```cpp
{
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);  // ref_count = 1
    std::shared_ptr<int> ptr2 = ptr1;                        // ref_count = 2
    ptr1.reset();                                            // ref_count = 1
}  // ptr2 destroyed, ref_count = 0, memory freed
```

**COM (Component Object Model):**
```cpp
interface IUnknown {
    virtual ULONG AddRef() = 0;
    virtual ULONG Release() = 0;
};

// Usage
IFoo* foo = CreateFoo();  // ref_count = 1
foo->AddRef();             // ref_count = 2
foo->Release();            // ref_count = 1
foo->Release();            // ref_count = 0, freed
```

### Mark and Sweep

Two-phase garbage collection: mark reachable objects, then sweep unreachable ones.

#### Algorithm

**Phase 1: Mark**
```
1. Start from root set (globals, stack variables, registers)
2. Traverse object graph, marking each reachable object
3. Use depth-first or breadth-first search
```

**Phase 2: Sweep**
```
1. Scan entire heap
2. Free unmarked objects
3. Reset marks for next collection
```

#### Visual Example

```
Initial State:
Root → [A] → [B] → [C]
       ↓
      [D]     [E]  [F] → [G]

Objects: A, B, C, D reachable from root
Objects: E, F, G unreachable (garbage)

After Mark Phase:
Root → [A]* → [B]* → [C]*
       ↓
      [D]*    [E]  [F] → [G]

(*= marked)

After Sweep Phase:
Root → [A] → [B] → [C]
       ↓
      [D]

E, F, G freed
```

#### Simple Implementation

```c
#define MARK_BIT 0x1

struct Object {
    struct Object* next;      // For linking in heap list
    unsigned flags;           // MARK_BIT stored here
    void* data;
    size_t size;
    struct Object** refs;     // Pointers to other objects
    size_t num_refs;
};

struct GC {
    struct Object* heap;      // All allocated objects
    struct Object** roots;    // Root set
    size_t num_roots;
};

// Mark phase: recursively mark reachable objects
void gc_mark(struct Object* obj) {
    if (!obj || (obj->flags & MARK_BIT)) {
        return;  // Already marked
    }

    obj->flags |= MARK_BIT;  // Mark this object

    // Recursively mark referenced objects
    for (size_t i = 0; i < obj->num_refs; i++) {
        gc_mark(obj->refs[i]);
    }
}

// Sweep phase: free unmarked objects
void gc_sweep(struct GC* gc) {
    struct Object** obj_ptr = &gc->heap;

    while (*obj_ptr) {
        struct Object* obj = *obj_ptr;

        if (!(obj->flags & MARK_BIT)) {
            // Unmarked - remove from list and free
            *obj_ptr = obj->next;
            free(obj->data);
            free(obj->refs);
            free(obj);
        } else {
            // Marked - clear mark for next cycle
            obj->flags &= ~MARK_BIT;
            obj_ptr = &obj->next;
        }
    }
}

// Full garbage collection
void gc_collect(struct GC* gc) {
    // Mark phase
    for (size_t i = 0; i < gc->num_roots; i++) {
        gc_mark(gc->roots[i]);
    }

    // Sweep phase
    gc_sweep(gc);
}
```

#### Iterative Marking (避免堆栈溢出)

Recursive marking can overflow stack for deep object graphs. Use iterative approach:

```c
void gc_mark_iterative(struct Object* root) {
    // Use explicit stack
    struct Object** stack = malloc(sizeof(struct Object*) * 1000);
    int top = 0;

    stack[top++] = root;

    while (top > 0) {
        struct Object* obj = stack[--top];

        if (!obj || (obj->flags & MARK_BIT)) {
            continue;
        }

        obj->flags |= MARK_BIT;

        // Push children onto stack
        for (size_t i = 0; i < obj->num_refs; i++) {
            if (top < 1000) {  // Prevent overflow
                stack[top++] = obj->refs[i];
            }
        }
    }

    free(stack);
}
```

#### Advantages

1. **Handles cycles**: Unreachable cycles are collected
2. **No overhead per assignment**: Unlike reference counting
3. **Simple conceptually**: Mark reachable, free unreachable

#### Disadvantages

1. **Stop-the-world pauses**: Must pause program during collection
2. **Unpredictable timing**: Collection happens when heap fills
3. **Memory overhead**: Need mark bits
4. **Fragmentation**: Freed objects leave gaps

#### Optimizations

**1. Tri-Color Marking** (see next section)

**2. Lazy Sweeping**
```c
// Don't sweep all at once
// Sweep incrementally during allocations
void* gc_alloc_with_lazy_sweep(struct GC* gc, size_t size) {
    // Sweep a few objects
    for (int i = 0; i < 10; i++) {
        if (gc->sweep_pos) {
            // Sweep one object
            gc->sweep_pos = gc->sweep_pos->next;
        }
    }

    // Then allocate
    return allocate(size);
}
```

**3. Generational Collection** (see later section)

#### When Mark-Sweep Runs

**Trigger 1: Heap Full**
```c
void* gc_alloc(struct GC* gc, size_t size) {
    void* ptr = try_allocate(size);

    if (!ptr) {
        // Out of memory - collect garbage
        gc_collect(gc);
        ptr = try_allocate(size);
    }

    return ptr;
}
```

**Trigger 2: Periodic**
```c
void main_loop() {
    static int alloc_count = 0;

    while (1) {
        do_work();

        if (++alloc_count > 1000) {
            gc_collect(&gc);
            alloc_count = 0;
        }
    }
}
```

**Trigger 3: Manual**
```c
// Explicit collection call
gc_collect(&gc);
```

### Tri-Color Marking

An incremental marking algorithm that allows GC work to be interleaved with program execution.

#### The Three Colors

**White**: Not yet visited; candidates for collection
**Gray**: Visited but children not yet scanned
**Black**: Visited and all children scanned

#### Algorithm

```
Initial:
- All objects are WHITE
- Roots are GRAY

While GRAY objects exist:
1. Pick a GRAY object
2. Mark it BLACK
3. Mark its WHITE children GRAY

After marking:
- BLACK objects are reachable (keep)
- WHITE objects are unreachable (collect)
```

#### Visual Example

```
Initial State:
ROOT → [A] → [B] → [C]
       ↓
      [D]     [E]

All objects WHITE

Step 1: Mark roots GRAY
ROOT → [A:GRAY] → [B:WHITE] → [C:WHITE]
       ↓
      [D:WHITE]    [E:WHITE]

Step 2: Process A (GRAY → BLACK, mark children GRAY)
ROOT → [A:BLACK] → [B:GRAY] → [C:WHITE]
       ↓
      [D:GRAY]     [E:WHITE]

Step 3: Process B (GRAY → BLACK, mark children GRAY)
ROOT → [A:BLACK] → [B:BLACK] → [C:GRAY]
       ↓
      [D:GRAY]     [E:WHITE]

Step 4: Process D (GRAY → BLACK, no children)
ROOT → [A:BLACK] → [B:BLACK] → [C:GRAY]
       ↓
      [D:BLACK]    [E:WHITE]

Step 5: Process C (GRAY → BLACK, no children)
ROOT → [A:BLACK] → [B:BLACK] → [C:BLACK]
       ↓
      [D:BLACK]    [E:WHITE]

Done! E is WHITE → collect it
```

#### Implementation

```c
enum Color { WHITE, GRAY, BLACK };

struct Object {
    enum Color color;
    void* data;
    struct Object** refs;
    size_t num_refs;
    struct Object* next;
};

struct GC {
    struct Object* all_objects;
    struct Object* gray_list;  // Work list
    struct Object** roots;
    size_t num_roots;
};

// Initialize all objects to WHITE
void gc_init(struct GC* gc) {
    for (struct Object* obj = gc->all_objects; obj; obj = obj->next) {
        obj->color = WHITE;
    }
    gc->gray_list = NULL;
}

// Add object to gray list
void gc_mark_gray(struct GC* gc, struct Object* obj) {
    if (obj->color == WHITE) {
        obj->color = GRAY;
        obj->next_gray = gc->gray_list;
        gc->gray_list = obj;
    }
}

// Process one gray object (incremental step)
void gc_process_one_gray(struct GC* gc) {
    if (!gc->gray_list) {
        return;  // No work to do
    }

    // Remove from gray list
    struct Object* obj = gc->gray_list;
    gc->gray_list = obj->next_gray;

    // Mark black
    obj->color = BLACK;

    // Mark children gray
    for (size_t i = 0; i < obj->num_refs; i++) {
        gc_mark_gray(gc, obj->refs[i]);
    }
}

// Full collection
void gc_collect(struct GC* gc) {
    // Initialize
    gc_init(gc);

    // Mark roots gray
    for (size_t i = 0; i < gc->num_roots; i++) {
        gc_mark_gray(gc, gc->roots[i]);
    }

    // Process all gray objects
    while (gc->gray_list) {
        gc_process_one_gray(gc);
    }

    // Sweep: free all WHITE objects
    struct Object** obj_ptr = &gc->all_objects;
    while (*obj_ptr) {
        if ((*obj_ptr)->color == WHITE) {
            struct Object* garbage = *obj_ptr;
            *obj_ptr = garbage->next;
            free(garbage);
        } else {
            obj_ptr = &(*obj_ptr)->next;
        }
    }
}

// Incremental collection (process N objects)
void gc_incremental_collect(struct GC* gc, int steps) {
    for (int i = 0; i < steps && gc->gray_list; i++) {
        gc_process_one_gray(gc);
    }
}
```

#### Incremental Collection

```c
void* gc_alloc(struct GC* gc, size_t size) {
    // Do a little GC work on each allocation
    if (gc->gc_in_progress) {
        gc_incremental_collect(gc, 10);  // Process 10 objects
    }

    void* ptr = allocate(size);

    if (!ptr) {
        // Start new GC cycle
        gc_start_collection(gc);
        ptr = allocate(size);
    }

    return ptr;
}
```

#### Write Barrier Problem

When program runs concurrently with incremental GC, need to track pointer updates:

```
Scenario:
1. A is BLACK (fully scanned)
2. B is WHITE (not yet visited)
3. C is GRAY (in progress)

Program executes: A.field = B

Problem: B might never be marked!
- A is BLACK (won't be rescanned)
- B is WHITE (not in gray list)
- After marking completes, B is still WHITE → incorrectly collected!
```

**Solution: Write Barrier**

```c
void object_set_field(struct Object* obj, size_t field, struct Object* value) {
    obj->refs[field] = value;

    // Write barrier
    if (obj->color == BLACK && value->color == WHITE) {
        // Re-mark object as GRAY
        gc_mark_gray(&gc, obj);
        // Or mark value GRAY:
        // gc_mark_gray(&gc, value);
    }
}
```

#### Advantages

1. **Incremental**: Can pause/resume marking
2. **Lower pause times**: Spread work over time
3. **Handles cycles**: Like regular mark-sweep

#### Disadvantages

1. **Write barrier overhead**: Every pointer update must be tracked
2. **Complexity**: More complex than simple mark-sweep
3. **Floating garbage**: Some garbage survives until next cycle

### Generational GC

Exploit the generational hypothesis: "Most objects die young."

#### Generational Hypothesis

**Observation:**
- 90%+ of objects die within a short time of allocation
- Long-lived objects tend to stay long-lived

**Implication:**
- Collect young objects frequently (fast)
- Collect old objects infrequently (slow but rare)

#### Multi-Generation Heap

```
┌─────────────────────────────────────────────────────┐
│                 Young Generation                     │
│  (Eden)  │  (Survivor 0)  │  (Survivor 1)           │
│  [new objects] [survived 1 GC] [survived 2+ GCs]    │
└─────────────────────────────────────────────────────┘
                    ↓ (promotion)
┌─────────────────────────────────────────────────────┐
│                 Old Generation                       │
│  [long-lived objects]                               │
└─────────────────────────────────────────────────────┘
                    ↓ (promotion)
┌─────────────────────────────────────────────────────┐
│            Permanent Generation (Java)               │
│  [class metadata, interned strings]                 │
└─────────────────────────────────────────────────────┘
```

#### Algorithm

**Minor GC (Young Generation):**
```
1. Mark live objects in young generation
2. Copy live objects to survivor space
3. Clear eden space
4. Promote old survivors to old generation
```

**Major GC (Old Generation):**
```
1. Mark live objects in entire heap
2. Sweep/compact old generation
3. Much slower, but rare
```

#### Example Implementation (Simplified)

```c
#define YOUNG_GEN_SIZE (1024 * 1024)  // 1 MB
#define OLD_GEN_SIZE (10 * 1024 * 1024)  // 10 MB

struct Object {
    int generation;  // 0 = young, 1 = old
    int age;         // Survived GC count
    void* data;
    struct Object** refs;
    size_t num_refs;
};

struct GenerationalGC {
    struct Object* young_gen;
    struct Object* old_gen;
    size_t young_size;
    size_t old_size;
};

void minor_gc(struct GenerationalGC* gc) {
    // Mark live objects in young generation
    struct Object* survivors = NULL;

    for (struct Object* obj = gc->young_gen; obj; obj = obj->next) {
        if (is_reachable(obj)) {
            obj->age++;

            if (obj->age > 3) {
                // Promote to old generation
                promote_to_old(gc, obj);
            } else {
                // Keep in young generation
                obj->next = survivors;
                survivors = obj;
            }
        } else {
            // Free
            free_object(obj);
        }
    }

    gc->young_gen = survivors;
}

void major_gc(struct GenerationalGC* gc) {
    // Full heap collection (slow)
    mark_and_sweep(gc->young_gen);
    mark_and_sweep(gc->old_gen);
}

void* gc_alloc(struct GenerationalGC* gc, size_t size) {
    // Try allocating in young generation
    void* ptr = allocate_in_young(gc, size);

    if (!ptr) {
        // Young generation full - minor GC
        minor_gc(gc);
        ptr = allocate_in_young(gc, size);
    }

    if (!ptr) {
        // Still no space - major GC
        major_gc(gc);
        ptr = allocate_in_young(gc, size);
    }

    return ptr;
}
```

#### Card Table for Cross-Generation References

Problem: Old objects might reference young objects. How to find roots for minor GC without scanning old generation?

**Solution: Card Table**

```
Old Generation divided into "cards" (e.g., 512-byte regions)

Card Table: [0][0][1][0][1][0][0][0]...
             ↑           ↑
        No refs    Has refs to young gen

When old object updated:
1. Mark corresponding card as "dirty"
2. During minor GC, only scan dirty cards
```

**Implementation:**
```c
#define CARD_SIZE 512
#define NUM_CARDS (OLD_GEN_SIZE / CARD_SIZE)

struct GenerationalGC {
    // ...
    unsigned char card_table[NUM_CARDS];  // 0 = clean, 1 = dirty
};

void write_barrier(void* old_obj, struct Object* value) {
    if (value->generation == 0) {  // Young object
        // Mark card dirty
        size_t card_index = ((char*)old_obj - old_gen_start) / CARD_SIZE;
        gc.card_table[card_index] = 1;
    }
}

void minor_gc_with_card_table(struct GenerationalGC* gc) {
    // Scan stack roots
    mark_from_roots();

    // Scan dirty cards in old generation
    for (size_t i = 0; i < NUM_CARDS; i++) {
        if (gc->card_table[i]) {
            void* card_start = old_gen_start + i * CARD_SIZE;
            scan_card_for_young_refs(card_start);
            gc->card_table[i] = 0;  // Clear dirty bit
        }
    }

    // Collect young generation
    collect_young_gen();
}
```

#### Performance Characteristics

**Minor GC:**
- Frequency: Very high (every few seconds)
- Pause time: Very low (< 10 ms)
- Throughput: High (most objects die young)

**Major GC:**
- Frequency: Low (every few minutes/hours)
- Pause time: High (100+ ms)
- Throughput: Lower (must scan entire heap)

**Example (JVM):**
```
Minor GC: 2-5 ms pause, every 1-10 seconds
Major GC: 100-500 ms pause, every 10-60 minutes
```

#### Advantages

1. **Fast minor GCs**: Only collect young generation
2. **Exploits generational hypothesis**: Most work on short-lived objects
3. **Lower average pause times**: Minor GCs are frequent but fast

#### Disadvantages

1. **Write barrier overhead**: Must track cross-generation pointers
2. **Complexity**: More complex than single-generation
3. **Promotion failures**: Can trigger full GC unexpectedly

### GC Tuning

Adjusting garbage collector parameters for optimal performance.

#### Key Metrics

**1. Throughput**
```
Throughput = Application Time / (Application Time + GC Time)

Example:
- Application runs 90 seconds
- GC runs 10 seconds
- Throughput = 90 / 100 = 90%
```

**2. Latency (Pause Time)**
```
Max pause time: Longest single GC pause
Average pause time: Mean of all GC pauses
99th percentile: 99% of pauses below this time
```

**3. Footprint (Memory Usage)**
```
Heap size
Live set size (reachable objects)
Memory overhead (GC metadata)
```

**Trade-offs:**
- Larger heap → Higher throughput, longer pauses
- Smaller heap → Lower throughput, shorter pauses, more frequent GC

#### Java GC Tuning

**Heap Size:**
```bash
# Initial and maximum heap size
java -Xms2g -Xmx4g MyApp

# Young generation size
java -Xmn1g MyApp

# Or ratio of young/old
java -XX:NewRatio=2 MyApp  # Old = 2 * Young
```

**GC Algorithm Selection:**
```bash
# Serial GC (single-threaded, low overhead)
java -XX:+UseSerialGC MyApp

# Parallel GC (multi-threaded, high throughput)
java -XX:+UseParallelGC MyApp

# CMS (Concurrent Mark Sweep, low latency)
java -XX:+UseConcMarkSweepGC MyApp

# G1 GC (Garbage First, balanced)
java -XX:+UseG1GC MyApp

# ZGC (ultra-low latency, JDK 11+)
java -XX:+UseZGC MyApp

# Shenandoah (low latency, JDK 12+)
java -XX:+UseShenandoahGC MyApp
```

**GC Logging:**
```bash
# Enable GC logging (JDK 8)
java -XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log MyApp

# Enable GC logging (JDK 9+)
java -Xlog:gc*:file=gc.log:time,uptime,level,tags MyApp
```

**Example GC Log Analysis:**
```
[GC (Allocation Failure) 2021-01-01T10:00:00.123+0000: 1.234:
  [ParNew: 614400K->68068K(614400K), 0.0924544 secs]
  614400K->68068K(2063104K), 0.0925372 secs]
  [Times: user=0.15 sys=0.01, real=0.09 secs]

Interpretation:
- Type: Minor GC (ParNew)
- Reason: Allocation Failure (young gen full)
- Young gen: 614400K → 68068K (89% freed!)
- Total heap: 614400K → 68068K
- Pause time: 92.5 ms
- CPU time: user=150ms, sys=10ms, real=90ms (parallelism ~1.7x)
```

#### Python GC Tuning

**Adjust Thresholds:**
```python
import gc

# Get current thresholds
print(gc.get_threshold())  # (700, 10, 10)

# Set new thresholds (threshold0, threshold1, threshold2)
gc.set_threshold(1000, 15, 15)

# threshold0: # of allocations before gen0 collection
# threshold1: # of gen0 collections before gen1 collection
# threshold2: # of gen1 collections before gen2 collection
```

**Disable/Enable GC:**
```python
# Disable automatic GC
gc.disable()

# Do work...

# Manually trigger collection
gc.collect()

# Re-enable automatic GC
gc.enable()
```

**Manual Collection Strategy:**
```python
import gc

def batch_process(items):
    gc.disable()  # Disable during processing

    for item in items:
        process(item)

    gc.collect()  # Collect once at end
    gc.enable()
```

#### Go GC Tuning

**GOGC Environment Variable:**
```bash
# Default: GOGC=100 (run GC when heap doubles)
# GOGC=200 (run GC when heap triples)
# GOGC=50 (run GC when heap grows 50%)
# GOGC=off (disable GC)

GOGC=200 ./myapp  # Less frequent GC, more memory
```

**Set Target Memory:**
```bash
# New in Go 1.19: set memory limit
GOMEMLIMIT=2GiB ./myapp
```

**Manual GC:**
```go
import "runtime"

func cleanup() {
    runtime.GC()  // Force garbage collection
}
```

**GC Tracing:**
```bash
# Print GC trace
GODEBUG=gctrace=1 ./myapp

# Example output:
# gc 1 @0.002s 5%: 0.015+0.85+0.003 ms clock, 0.12+0.12/0.70/0.015+0.025 ms cpu, 4->4->0 MB, 5 MB goal, 8 P
#
# Interpretation:
# - GC #1
# - At 0.002 seconds
# - 5% CPU time in GC
# - Heap: 4 MB → 4 MB → 0 MB (before GC, after mark, after sweep)
# - Goal: 5 MB (next GC trigger)
# - 8 P (processors)
```

#### Tuning Strategy

**1. Measure First**
```
- Profile application
- Identify GC overhead
- Measure pause times
- Check memory usage
```

**2. Set Goals**
```
Throughput-oriented:
- Maximize application CPU time
- Accept longer pause times
- Use Parallel GC (Java) or larger heap

Latency-oriented:
- Minimize pause times
- Accept lower throughput
- Use CMS/G1/ZGC (Java) or smaller heap
```

**3. Tune Incrementally**
```
- Change one parameter at a time
- Measure impact
- Iterate
```

**4. Common Tuning Patterns**

**Pattern 1: High Throughput**
```bash
# Java
java -Xms8g -Xmx8g -XX:+UseParallelGC -XX:ParallelGCThreads=8 MyApp

# Large heap, parallel collection
```

**Pattern 2: Low Latency**
```bash
# Java
java -Xmx4g -XX:+UseZGC -XX:MaxGCPauseMillis=10 MyApp

# ZGC for sub-10ms pauses
```

**Pattern 3: Batch Processing**
```python
# Python: disable GC during batch, collect after
gc.disable()
process_large_dataset()
gc.collect()
gc.enable()
```

### GC Pauses

Understanding and minimizing garbage collection pauses.

#### Types of Pauses

**1. Stop-the-World (STW)**
```
Application threads:  ████░░░░░░░░████████
GC thread:            ░░░░████████░░░░░░░░
                          ↑
                      STW pause
```

All application threads stopped during GC.

**2. Concurrent**
```
Application threads:  ████████████████████
GC thread:            ░░░░████████████░░░░
                          ↑
                      Running concurrently
```

GC runs while application continues (with write barriers).

**3. Incremental**
```
Application threads:  ██░█░█░█░█░█░█░█░███
GC thread:            ░░█░█░█░█░█░█░█░█░░░
                        ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
                      Short pauses
```

GC work interleaved with application.

#### Pause Time Analysis

**Measuring Pauses (Java):**
```bash
# GC log shows pause times
java -Xlog:gc:file=gc.log MyApp

# Analyze with GCViewer or similar tool
```

**Example GC Pause Distribution:**
```
P50 (median): 10 ms
P90: 50 ms
P99: 200 ms
P99.9: 500 ms
Max: 2000 ms
```

**Interpreting:**
- 50% of pauses ≤ 10 ms (good)
- 90% of pauses ≤ 50 ms (acceptable)
- 1% of pauses > 200 ms (may be problematic)
- Max pause of 2 seconds (bad for latency-sensitive apps)

#### Reducing Pause Times

**Strategy 1: Use Concurrent Collector**

**Java CMS (Concurrent Mark Sweep):**
```bash
java -XX:+UseConcMarkSweepGC -XX:+CMSIncrementalMode MyApp
```

**Phases:**
```
1. Initial Mark (STW, short): Mark roots
2. Concurrent Mark: Mark reachable objects
3. Remark (STW, short): Catch changes during concurrent mark
4. Concurrent Sweep: Free garbage

STW pauses are short (10-100 ms)
```

**Java G1 (Garbage First):**
```bash
java -XX:+UseG1GC -XX:MaxGCPauseMillis=100 MyApp
```

**Characteristics:**
- Divides heap into regions
- Collects regions with most garbage first
- Predictable pause times
- Target: ~100 ms pauses

**Java ZGC:**
```bash
java -XX:+UseZGC MyApp
```

**Characteristics:**
- Sub-10ms pause times (even for 1+ TB heaps!)
- Concurrent compaction
- Colored pointers for tracking

**Strategy 2: Reduce Heap Size**

```bash
# Smaller heap = shorter GC pauses
# But more frequent GC

# Before: 8 GB heap, 500 ms pauses
java -Xmx8g MyApp

# After: 2 GB heap, 100 ms pauses (but 4x more frequent)
java -Xmx2g MyApp
```

**Strategy 3: Increase Young Generation Size**

```bash
# Larger young gen = less frequent minor GCs
java -Xmn2g MyApp

# But each minor GC takes longer
```

**Strategy 4: Tune GC Threads**

```bash
# More threads = shorter pause (if CPU available)
java -XX:ParallelGCThreads=8 MyApp

# Balance: too many threads causes contention
```

**Strategy 5: Avoid Finalizers**

```java
// BAD: Finalizers slow down GC
class BadResource {
    @Override
    protected void finalize() {  // Don't use!
        cleanup();
    }
}

// GOOD: Explicit cleanup
class GoodResource implements AutoCloseable {
    @Override
    public void close() {
        cleanup();
    }
}

try (GoodResource r = new GoodResource()) {
    // Use resource
}  // Automatically cleaned up
```

**Strategy 6: Object Pooling**

```java
// Reuse objects instead of allocating new ones
class ObjectPool<T> {
    private Queue<T> pool = new ConcurrentLinkedQueue<>();

    public T acquire() {
        T obj = pool.poll();
        return obj != null ? obj : createNew();
    }

    public void release(T obj) {
        reset(obj);
        pool.offer(obj);
    }
}

// Reduces allocation rate → less GC pressure
```

#### Real-World Example

**Before Tuning:**
```
Application: Latency-sensitive web service
Heap: 4 GB
GC: Parallel GC
Pause times: P99 = 800 ms (too high!)
Throughput: 95%
```

**After Tuning:**
```bash
# Switch to G1 with pause time goal
java -Xms4g -Xmx4g \
     -XX:+UseG1GC \
     -XX:MaxGCPauseMillis=50 \
     -XX:G1HeapRegionSize=16m \
     MyApp
```

**Results:**
```
Pause times: P99 = 45 ms (improved!)
Throughput: 92% (slight decrease, acceptable)
```

---

## Manual Memory Management

Explicit allocation and deallocation of memory by the programmer.

### malloc/free in C

#### Basic Usage

```c
#include <stdlib.h>

// Allocate memory
int* ptr = malloc(sizeof(int) * 10);
if (ptr == NULL) {
    // Handle allocation failure
    fprintf(stderr, "Out of memory\n");
    return -1;
}

// Use memory
for (int i = 0; i < 10; i++) {
    ptr[i] = i * i;
}

// Free memory
free(ptr);
ptr = NULL;  // Best practice: nullify after free
```

#### Common Patterns

**Pattern 1: Dynamic Strings**
```c
char* create_greeting(const char* name) {
    size_t len = strlen(name) + strlen("Hello, ") + 2;  // +2 for "!\0"
    char* greeting = malloc(len);
    if (!greeting) return NULL;

    sprintf(greeting, "Hello, %s!", name);
    return greeting;  // Caller must free!
}

// Usage
char* msg = create_greeting("Alice");
if (msg) {
    printf("%s\n", msg);
    free(msg);
}
```

**Pattern 2: Dynamic Arrays**
```c
struct DynArray {
    int* data;
    size_t size;
    size_t capacity;
};

void array_init(struct DynArray* arr) {
    arr->data = NULL;
    arr->size = 0;
    arr->capacity = 0;
}

int array_push(struct DynArray* arr, int value) {
    if (arr->size >= arr->capacity) {
        size_t new_cap = arr->capacity ? arr->capacity * 2 : 4;
        int* new_data = realloc(arr->data, new_cap * sizeof(int));
        if (!new_data) return -1;  // Allocation failed

        arr->data = new_data;
        arr->capacity = new_cap;
    }

    arr->data[arr->size++] = value;
    return 0;
}

void array_destroy(struct DynArray* arr) {
    free(arr->data);
    arr->data = NULL;
    arr->size = arr->capacity = 0;
}
```

**Pattern 3: Structures with Pointers**
```c
struct Person {
    char* name;
    char* email;
    int age;
};

struct Person* person_create(const char* name, const char* email, int age) {
    struct Person* p = malloc(sizeof(struct Person));
    if (!p) return NULL;

    p->name = strdup(name);    // strdup = malloc + strcpy
    p->email = strdup(email);

    if (!p->name || !p->email) {
        free(p->name);
        free(p->email);
        free(p);
        return NULL;
    }

    p->age = age;
    return p;
}

void person_destroy(struct Person* p) {
    if (p) {
        free(p->name);
        free(p->email);
        free(p);
    }
}
```

#### Memory Allocation Functions

**malloc() vs calloc() vs realloc():**

```c
// malloc: uninitialized memory
int* a = malloc(10 * sizeof(int));
// a[0] has garbage value

// calloc: zero-initialized memory
int* b = calloc(10, sizeof(int));
// b[0] == 0

// realloc: resize existing allocation
a = realloc(a, 20 * sizeof(int));
// First 10 elements preserved, next 10 uninitialized
```

**Performance:**
```c
// Benchmark: malloc vs calloc
clock_t start;

start = clock();
for (int i = 0; i < 100000; i++) {
    int* p = malloc(1000 * sizeof(int));
    free(p);
}
printf("malloc: %f s\n", (double)(clock() - start) / CLOCKS_PER_SEC);

start = clock();
for (int i = 0; i < 100000; i++) {
    int* p = calloc(1000, sizeof(int));
    free(p);
}
printf("calloc: %f s\n", (double)(clock() - start) / CLOCKS_PER_SEC);

// calloc typically 2-3x slower due to zeroing
```

#### Common Mistakes

**Mistake 1: Memory Leak**
```c
// BAD: Memory leak
void bad_function() {
    char* ptr = malloc(1000);
    // Forgot to free!
}  // ptr goes out of scope, memory leaked

// GOOD
void good_function() {
    char* ptr = malloc(1000);
    // Use ptr...
    free(ptr);
}
```

**Mistake 2: Use After Free**
```c
// BAD: Use after free
int* ptr = malloc(sizeof(int));
*ptr = 42;
free(ptr);
printf("%d\n", *ptr);  // Undefined behavior!

// GOOD
int* ptr = malloc(sizeof(int));
*ptr = 42;
printf("%d\n", *ptr);
free(ptr);
ptr = NULL;  // Nullify to catch errors
```

**Mistake 3: Double Free**
```c
// BAD: Double free
int* ptr = malloc(sizeof(int));
free(ptr);
free(ptr);  // Undefined behavior!

// GOOD
int* ptr = malloc(sizeof(int));
free(ptr);
ptr = NULL;
// free(NULL) is safe (no-op)
```

**Mistake 4: Incorrect Size**
```c
// BAD: Wrong size
int* arr = malloc(10);  // Only 10 bytes, not 10 ints!

// GOOD
int* arr = malloc(10 * sizeof(int));
// Or
int* arr = malloc(sizeof(int[10]));
```

**Mistake 5: Not Checking Return Value**
```c
// BAD: No error checking
int* ptr = malloc(1000000000);
*ptr = 42;  // Crash if malloc failed!

// GOOD
int* ptr = malloc(1000000000);
if (!ptr) {
    fprintf(stderr, "Allocation failed\n");
    return -1;
}
*ptr = 42;
```

### new/delete in C++

#### Basic Usage

```cpp
// Single object
int* ptr = new int(42);
delete ptr;

// Array
int* arr = new int[10];
delete[] arr;  // Note: delete[], not delete!

// With constructor
class Person {
public:
    Person(std::string name) : name(name) {}
    ~Person() { std::cout << "Destroying " << name << "\n"; }
private:
    std::string name;
};

Person* p = new Person("Alice");
delete p;  // Calls destructor automatically
```

#### new vs malloc

| Feature | new | malloc |
|---------|-----|--------|
| Type | Operator | Function |
| Returns | Typed pointer | void* |
| Size | Automatic | Manual calculation |
| Initialization | Calls constructor | No initialization |
| Failure | Throws exception | Returns NULL |
| Overloadable | Yes | No |

```cpp
// new: type-safe, calls constructor
std::string* s1 = new std::string("Hello");

// malloc: type-unsafe, no constructor
std::string* s2 = (std::string*)malloc(sizeof(std::string));
// BUG: s2 not initialized! (no constructor called)
```

#### Placement new

Construct object at specific memory address:

```cpp
#include <new>

// Allocate raw memory
void* buffer = malloc(sizeof(std::string));

// Construct object in that memory
std::string* s = new (buffer) std::string("Hello");

// Use object
std::cout << *s << "\n";

// Manually call destructor
s->~string();

// Free memory
free(buffer);
```

**Use case: Memory pools**
```cpp
class ObjectPool {
    char buffer[1000 * sizeof(MyClass)];

public:
    MyClass* allocate() {
        void* ptr = get_free_slot();
        return new (ptr) MyClass();  // Placement new
    }

    void deallocate(MyClass* obj) {
        obj->~MyClass();  // Manual destructor call
        mark_slot_free(obj);
    }
};
```

#### Array new/delete

```cpp
// Allocate array
int* arr = new int[10];

// MUST use delete[]
delete[] arr;  // Correct

// BAD: Using delete instead of delete[]
delete arr;  // Undefined behavior! Memory corruption!
```

**Why separate delete[]?**
```cpp
class MyClass {
public:
    MyClass() { std::cout << "Constructor\n"; }
    ~MyClass() { std::cout << "Destructor\n"; }
};

MyClass* arr = new MyClass[3];
// Calls constructor 3 times

delete[] arr;
// Calls destructor 3 times

delete arr;
// Only calls destructor once! Other 2 objects not destroyed!
```

#### nothrow new

```cpp
// Default: throws std::bad_alloc on failure
try {
    int* ptr = new int[1000000000000];  // Huge allocation
} catch (std::bad_alloc& e) {
    std::cerr << "Allocation failed: " << e.what() << "\n";
}

// nothrow: returns nullptr on failure (like malloc)
int* ptr = new (std::nothrow) int[1000000000000];
if (!ptr) {
    std::cerr << "Allocation failed\n";
}
```

#### Custom new/delete Operators

```cpp
class MyClass {
public:
    // Custom new operator
    void* operator new(size_t size) {
        std::cout << "Custom new: " << size << " bytes\n";
        void* ptr = ::operator new(size);  // Call global new
        return ptr;
    }

    // Custom delete operator
    void operator delete(void* ptr) {
        std::cout << "Custom delete\n";
        ::operator delete(ptr);  // Call global delete
    }
};

MyClass* obj = new MyClass();  // Calls MyClass::operator new
delete obj;                     // Calls MyClass::operator delete
```

**Use case: Tracking allocations**
```cpp
class Tracked {
    static size_t allocation_count;

public:
    void* operator new(size_t size) {
        allocation_count++;
        return ::operator new(size);
    }

    void operator delete(void* ptr) {
        allocation_count--;
        ::operator delete(ptr);
    }

    static size_t get_allocation_count() {
        return allocation_count;
    }
};

size_t Tracked::allocation_count = 0;
```

### Memory Leak Detection

#### Valgrind (Linux)

**Installation:**
```bash
sudo apt-get install valgrind
```

**Basic Usage:**
```bash
# Compile with debug symbols
gcc -g -o myapp myapp.c

# Run with Valgrind
valgrind --leak-check=full ./myapp
```

**Example Output:**
```
==12345== HEAP SUMMARY:
==12345==     in use at exit: 1,000 bytes in 1 blocks
==12345==   total heap usage: 2 allocs, 1 frees, 2,000 bytes allocated
==12345==
==12345== 1,000 bytes in 1 blocks are definitely lost in loss record 1 of 1
==12345==    at 0x4C2DB8F: malloc (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x40057E: main (myapp.c:10)
==12345==
==12345== LEAK SUMMARY:
==12345==    definitely lost: 1,000 bytes in 1 blocks
==12345==    indirectly lost: 0 bytes in 0 blocks
==12345==      possibly lost: 0 bytes in 0 blocks
==12345==    still reachable: 0 bytes in 0 blocks
==12345==         suppressed: 0 bytes in 0 blocks
```

**Leak Categories:**
- **Definitely lost**: No pointers to block
- **Indirectly lost**: Lost through lost container
- **Possibly lost**: Pointer exists but not to start of block
- **Still reachable**: Pointer still exists (not necessarily a leak)

**Advanced Options:**
```bash
# Track all allocations (slow but thorough)
valgrind --leak-check=full --show-leak-kinds=all ./myapp

# Generate suppression file for false positives
valgrind --gen-suppressions=all ./myapp 2>supp.txt

# Use suppression file
valgrind --suppressions=supp.txt ./myapp
```

#### AddressSanitizer (ASan)

Compiler-based tool for detecting memory errors.

**Compilation:**
```bash
# GCC/Clang
gcc -fsanitize=address -g -o myapp myapp.c

# Run normally (no special tool needed)
./myapp
```

**Detects:**
- Heap buffer overflow
- Stack buffer overflow
- Use-after-free
- Use-after-return
- Use-after-scope
- Double-free
- Memory leaks

**Example Output:**
```
=================================================================
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x60300000eff0
READ of size 4 at 0x60300000eff0 thread T0
    #0 0x400b42 in main myapp.c:15
    #1 0x7f8b7c8c3b96 in __libc_start_main
    #2 0x400a09 in _start

0x60300000eff0 is located 0 bytes inside of 4-byte region [0x60300000eff0,0x60300000eff4)
freed by thread T0 here:
    #0 0x7f8b7cc63537 in __interceptor_free
    #1 0x400b2d in main myapp.c:14
```

**Advantages over Valgrind:**
- Much faster (2-3x slowdown vs 20-50x)
- Catches more types of errors
- Works with multithreaded code better

**Disadvantages:**
- Requires recompilation
- Increases binary size
- May not catch all leaks

#### LeakSanitizer (LSan)

Part of AddressSanitizer, focused on leak detection.

```bash
# Enable leak detection (included with ASan)
gcc -fsanitize=address -g -o myapp myapp.c

# Or use LeakSanitizer standalone
gcc -fsanitize=leak -g -o myapp myapp.c

./myapp
```

**Suppress false positives:**
```c
// In code
const char* __lsan_default_suppressions() {
    return "leak:some_function\n";
}

// Or via environment variable
LSAN_OPTIONS=suppressions=supp.txt ./myapp
```

#### Manual Leak Tracking

**Simple Reference Counting:**
```c
#ifdef DEBUG_MEMORY
static size_t alloc_count = 0;
static size_t free_count = 0;

void* debug_malloc(size_t size, const char* file, int line) {
    void* ptr = malloc(size);
    if (ptr) {
        alloc_count++;
        printf("[ALLOC] %p (%zu bytes) at %s:%d\n", ptr, size, file, line);
    }
    return ptr;
}

void debug_free(void* ptr, const char* file, int line) {
    if (ptr) {
        free_count++;
        printf("[FREE] %p at %s:%d\n", ptr, file, line);
    }
    free(ptr);
}

#define malloc(size) debug_malloc(size, __FILE__, __LINE__)
#define free(ptr) debug_free(ptr, __FILE__, __LINE__)

void print_leak_summary() {
    printf("Allocations: %zu\n", alloc_count);
    printf("Frees: %zu\n", free_count);
    printf("Leaks: %zu\n", alloc_count - free_count);
}
#endif
```

**Allocation Tracking Table:**
```c
#define MAX_ALLOCATIONS 10000

struct Allocation {
    void* ptr;
    size_t size;
    const char* file;
    int line;
};

static struct Allocation allocations[MAX_ALLOCATIONS];
static size_t num_allocations = 0;

void track_allocation(void* ptr, size_t size, const char* file, int line) {
    if (num_allocations < MAX_ALLOCATIONS) {
        allocations[num_allocations++] = (struct Allocation){
            .ptr = ptr,
            .size = size,
            .file = file,
            .line = line
        };
    }
}

void untrack_allocation(void* ptr) {
    for (size_t i = 0; i < num_allocations; i++) {
        if (allocations[i].ptr == ptr) {
            allocations[i] = allocations[--num_allocations];
            return;
        }
    }
    fprintf(stderr, "ERROR: Free of untracked pointer %p\n", ptr);
}

void print_leaks() {
    printf("=== Memory Leaks ===\n");
    for (size_t i = 0; i < num_allocations; i++) {
        printf("Leak: %zu bytes at %s:%d\n",
               allocations[i].size,
               allocations[i].file,
               allocations[i].line);
    }
}
```

### Use-After-Free Bugs

Accessing memory after it has been freed.

#### Example

```c
int* ptr = malloc(sizeof(int));
*ptr = 42;
free(ptr);

// Use-after-free!
printf("%d\n", *ptr);  // Undefined behavior
*ptr = 100;            // Undefined behavior (likely crash)
```

#### Why It's Dangerous

**Scenario 1: Memory Reused**
```c
int* ptr1 = malloc(sizeof(int));
*ptr1 = 42;
free(ptr1);

// Another allocation reuses the same memory
char* ptr2 = malloc(sizeof(char) * 100);
strcpy(ptr2, "Hello");

// Use-after-free: corrupts ptr2!
*ptr1 = 100;

printf("%s\n", ptr2);  // Might print garbage
```

**Scenario 2: Security Vulnerability**
```c
struct User {
    char name[32];
    int is_admin;
};

struct User* user = malloc(sizeof(struct User));
strcpy(user->name, "Alice");
user->is_admin = 0;
free(user);

// Attacker allocates at same address
char* exploit = malloc(sizeof(struct User));
memset(exploit, 1, sizeof(struct User));  // Set is_admin = 1

// Use-after-free: treats exploit as user
if (user->is_admin) {
    printf("Admin access granted!\n");  // Security breach!
}
```

#### Detection with AddressSanitizer

```c
#include <stdlib.h>

int main() {
    int* ptr = malloc(sizeof(int));
    *ptr = 42;
    free(ptr);

    *ptr = 100;  // Use-after-free

    return 0;
}
```

```bash
$ gcc -fsanitize=address -g -o test test.c
$ ./test

=================================================================
==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x60300000eff0
WRITE of size 4 at 0x60300000eff0 thread T0
    #0 0x400b95 in main test.c:8

0x60300000eff0 is located 0 bytes inside of 4-byte region
freed by thread T0 here:
    #0 0x7f0b7cc63537 in __interceptor_free
    #1 0x400b80 in main test.c:7
```

#### Prevention

**1. Nullify After Free**
```c
int* ptr = malloc(sizeof(int));
// Use ptr...
free(ptr);
ptr = NULL;  // Further access will crash (better than corruption)

if (ptr) {
    *ptr = 100;  // Won't execute
}
```

**2. Use Wrapper Functions**
```c
#define SAFE_FREE(ptr) do { free(ptr); (ptr) = NULL; } while(0)

int* ptr = malloc(sizeof(int));
SAFE_FREE(ptr);  // Frees and nullifies

*ptr = 100;  // Crash (detectable) instead of corruption
```

**3. Smart Pointers (C++)**
```cpp
{
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    // Use ptr...
}  // Automatically freed, ptr no longer accessible
```

**4. Ownership Tracking**
```c
enum State { VALID, FREED };

struct TrackedPointer {
    void* ptr;
    enum State state;
};

struct TrackedPointer* create_tracked(size_t size) {
    struct TrackedPointer* tp = malloc(sizeof(struct TrackedPointer));
    tp->ptr = malloc(size);
    tp->state = VALID;
    return tp;
}

void* get_ptr(struct TrackedPointer* tp) {
    assert(tp->state == VALID && "Use-after-free detected!");
    return tp->ptr;
}

void free_tracked(struct TrackedPointer* tp) {
    assert(tp->state == VALID && "Double-free detected!");
    free(tp->ptr);
    tp->state = FREED;
}
```

### Double-Free Errors

Calling `free()` twice on the same pointer.

#### Example

```c
int* ptr = malloc(sizeof(int));
free(ptr);
free(ptr);  // Double-free! Undefined behavior
```

#### Why It's Dangerous

**Heap Corruption:**
```c
int* a = malloc(100);
int* b = malloc(100);
free(a);
free(a);  // Double-free corrupts heap metadata

int* c = malloc(100);  // May crash or return invalid pointer
```

**Exploitability:**
- Attackers can trigger double-free to corrupt heap
- Can lead to arbitrary code execution
- Common in security vulnerabilities

#### Detection

**AddressSanitizer:**
```c
int main() {
    int* ptr = malloc(sizeof(int));
    free(ptr);
    free(ptr);  // Double-free
    return 0;
}
```

```bash
$ gcc -fsanitize=address -g -o test test.c
$ ./test

=================================================================
==12345==ERROR: AddressSanitizer: attempting double-free on 0x60300000eff0
    #0 0x7f0b7cc63537 in __interceptor_free
    #1 0x400b95 in main test.c:5
```

**Valgrind:**
```bash
$ valgrind ./test

==12345== Invalid free() / delete / delete[] / realloc()
==12345==    at 0x4C2EDEB: free (in /usr/lib/valgrind/vgpreload_memcheck-amd64-linux.so)
==12345==    by 0x40057E: main (test.c:5)
==12345==  Address 0x5203040 is 0 bytes inside a block of size 4 free'd
```

#### Prevention

**1. Nullify After Free**
```c
int* ptr = malloc(sizeof(int));
free(ptr);
ptr = NULL;

free(ptr);  // Safe: free(NULL) is a no-op
```

**2. Safe Free Macro**
```c
#define SAFE_FREE(ptr) do { \
    free(ptr); \
    (ptr) = NULL; \
} while(0)

int* ptr = malloc(sizeof(int));
SAFE_FREE(ptr);
SAFE_FREE(ptr);  // Safe (second call frees NULL)
```

**3. Ownership Pattern**
```c
struct Resource {
    void* data;
    int owned;  // 1 if we own it, 0 if transferred
};

void resource_free(struct Resource* r) {
    if (r->owned) {
        free(r->data);
        r->owned = 0;
    }
}

// Transfer ownership
void resource_transfer(struct Resource* from, struct Resource* to) {
    to->data = from->data;
    to->owned = 1;
    from->owned = 0;  // No longer owns it
}
```

**4. RAII in C++**
```cpp
{
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    // Impossible to double-free with unique_ptr
}  // Automatically freed once
```

---

## Smart Pointers (C++)

Automatic memory management through RAII (Resource Acquisition Is Initialization).

### unique_ptr

Exclusive ownership smart pointer - only one unique_ptr can own a resource.

#### Basic Usage

```cpp
#include <memory>

// Create unique_ptr
std::unique_ptr<int> ptr1(new int(42));
// Or (preferred):
std::unique_ptr<int> ptr2 = std::make_unique<int>(42);

// Access
*ptr2 = 100;
std::cout << *ptr2 << "\n";  // 100

// Automatic cleanup when ptr2 goes out of scope
```

#### Arrays

```cpp
// Array unique_ptr
std::unique_ptr<int[]> arr = std::make_unique<int[]>(10);

// Access elements
arr[0] = 1;
arr[1] = 2;

// Automatically deletes with delete[], not delete
```

#### Move Semantics

```cpp
// unique_ptr cannot be copied (deleted copy constructor)
std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
// std::unique_ptr<int> ptr2 = ptr1;  // ERROR: cannot copy

// But can be moved (transfers ownership)
std::unique_ptr<int> ptr2 = std::move(ptr1);
// Now ptr1 is nullptr, ptr2 owns the resource
```

#### Return from Function

```cpp
std::unique_ptr<MyClass> create_object() {
    std::unique_ptr<MyClass> ptr = std::make_unique<MyClass>();
    // ...
    return ptr;  // Move semantics (no copy)
}

// Caller receives ownership
std::unique_ptr<MyClass> obj = create_object();
```

#### Custom Deleters

```cpp
// Custom deleter for FILE*
auto file_deleter = [](FILE* f) {
    if (f) fclose(f);
};

std::unique_ptr<FILE, decltype(file_deleter)> file(
    fopen("test.txt", "r"),
    file_deleter
);

// file automatically closed when unique_ptr destroyed

// Or with function pointer
void close_file(FILE* f) {
    if (f) fclose(f);
}

std::unique_ptr<FILE, void(*)(FILE*)> file2(
    fopen("test.txt", "r"),
    close_file
);
```

#### Release Ownership

```cpp
std::unique_ptr<int> ptr = std::make_unique<int>(42);

// Release ownership (returns raw pointer, unique_ptr becomes null)
int* raw = ptr.release();

// Now we're responsible for deletion
delete raw;
```

#### Reset

```cpp
std::unique_ptr<int> ptr = std::make_unique<int>(42);

// Delete current and manage new object
ptr.reset(new int(100));

// Or delete current and become null
ptr.reset();
// ptr is now nullptr
```

#### Advantages

- Zero overhead (same size as raw pointer)
- Automatic cleanup
- Move-only (clear ownership semantics)
- Type-safe
- Works with arrays

#### Use Cases

```cpp
// 1. Function-local resources
void process_file(const std::string& filename) {
    std::unique_ptr<File> file = open_file(filename);
    // Process file...
    // Automatic cleanup even if exception thrown
}

// 2. Class members (exclusive ownership)
class Widget {
    std::unique_ptr<Impl> pImpl;  // Pimpl idiom
public:
    Widget() : pImpl(std::make_unique<Impl>()) {}
    // Compiler-generated destructor automatically deletes pImpl
};

// 3. Factory functions
std::unique_ptr<Shape> create_shape(ShapeType type) {
    switch (type) {
        case CIRCLE: return std::make_unique<Circle>();
        case SQUARE: return std::make_unique<Square>();
    }
}
```

### shared_ptr

Shared ownership smart pointer - multiple shared_ptrs can own the same resource.

#### Basic Usage

```cpp
std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
std::cout << "Count: " << ptr1.use_count() << "\n";  // 1

{
    std::shared_ptr<int> ptr2 = ptr1;  // Copying allowed
    std::cout << "Count: " << ptr1.use_count() << "\n";  // 2
    *ptr2 = 100;
}  // ptr2 destroyed, count decrements

std::cout << "Count: " << ptr1.use_count() << "\n";  // 1
std::cout << "*ptr1: " << *ptr1 << "\n";  // 100

// When last shared_ptr destroyed, resource deleted
```

#### Reference Counting

```cpp
std::shared_ptr<int> ptr1 = std::make_shared<int>(42);  // ref_count = 1

std::shared_ptr<int> ptr2 = ptr1;                        // ref_count = 2
std::shared_ptr<int> ptr3 = ptr2;                        // ref_count = 3

ptr1.reset();                                            // ref_count = 2
ptr2 = nullptr;                                          // ref_count = 1
// Resource still alive (ptr3 still owns it)

ptr3.reset();                                            // ref_count = 0, deleted!
```

#### make_shared vs Constructor

```cpp
// Preferred: make_shared (one allocation)
std::shared_ptr<MyClass> ptr1 = std::make_shared<MyClass>(args);
// Allocates: [control block][MyClass object] in one block

// Not preferred: constructor (two allocations)
std::shared_ptr<MyClass> ptr2(new MyClass(args));
// Allocates: [MyClass object] and separately [control block]
```

**Performance difference:**
- `make_shared`: 1 allocation, better cache locality
- Constructor: 2 allocations, extra overhead

#### Circular Reference Problem

```cpp
struct Node {
    std::shared_ptr<Node> next;
    ~Node() { std::cout << "Destructor called\n"; }
};

{
    std::shared_ptr<Node> node1 = std::make_shared<Node>();
    std::shared_ptr<Node> node2 = std::make_shared<Node>();

    node1->next = node2;  // node2 ref_count = 2
    node2->next = node1;  // node1 ref_count = 2
}
// Both go out of scope, but ref_count still > 0!
// MEMORY LEAK! Destructors never called!
```

**Solution: Use weak_ptr (see next section)**

#### Thread-Safety

```cpp
// Reference count is thread-safe
std::shared_ptr<int> global_ptr = std::make_shared<int>(42);

void thread1() {
    std::shared_ptr<int> local = global_ptr;  // Thread-safe increment
}

void thread2() {
    std::shared_ptr<int> local = global_ptr;  // Thread-safe increment
}

// But the pointed-to object is NOT automatically thread-safe
void thread3() {
    *global_ptr = 100;  // Data race if thread4 runs concurrently!
}

void thread4() {
    *global_ptr = 200;  // Data race!
}
```

#### Custom Deleters

```cpp
std::shared_ptr<FILE> file(
    fopen("test.txt", "r"),
    [](FILE* f) { if (f) fclose(f); }
);

// Or with std::function
std::shared_ptr<Connection> conn(
    connect_to_server(),
    [](Connection* c) { disconnect(c); }
);
```

#### Aliasing Constructor

```cpp
struct Foo {
    int x;
    int y;
};

std::shared_ptr<Foo> foo = std::make_shared<Foo>();

// Aliasing: share ownership of foo, but point to foo->x
std::shared_ptr<int> x_ptr(foo, &foo->x);

// x_ptr.use_count() == 2
// foo won't be deleted until both foo and x_ptr are destroyed
```

#### Use Cases

```cpp
// 1. Shared resources
class ResourceManager {
    std::shared_ptr<Database> db;
public:
    std::shared_ptr<Database> get_database() {
        return db;  // Share ownership
    }
};

// 2. Observer pattern
class Subject {
    std::vector<std::shared_ptr<Observer>> observers;
public:
    void attach(std::shared_ptr<Observer> obs) {
        observers.push_back(obs);
    }
};

// 3. Cache with shared ownership
class Cache {
    std::map<std::string, std::shared_ptr<Data>> cache;
public:
    std::shared_ptr<Data> get(const std::string& key) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;  // Share cached data
        }
        return nullptr;
    }
};
```

### weak_ptr

Non-owning smart pointer that observes a shared_ptr without increasing ref count.

#### Basic Usage

```cpp
std::shared_ptr<int> sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;  // Doesn't increase ref count

std::cout << sp.use_count() << "\n";  // 1 (weak_ptr doesn't count)

// weak_ptr cannot access object directly
// *wp; // ERROR

// Must convert to shared_ptr first
if (std::shared_ptr<int> sp2 = wp.lock()) {
    // Object still alive
    std::cout << *sp2 << "\n";  // 42
    std::cout << sp.use_count() << "\n";  // 2
} else {
    // Object was deleted
    std::cout << "Object expired\n";
}
```

#### Breaking Circular References

```cpp
struct Node {
    std::shared_ptr<Node> next;     // Strong reference
    std::weak_ptr<Node> prev;       // Weak reference (breaks cycle)

    ~Node() { std::cout << "Destructor called\n"; }
};

{
    std::shared_ptr<Node> node1 = std::make_shared<Node>();
    std::shared_ptr<Node> node2 = std::make_shared<Node>();

    node1->next = node2;  // node2 ref_count = 2
    node2->prev = node1;  // node1 ref_count still 1 (weak_ptr doesn't count)
}
// node1 ref_count = 0 → deleted
// node2 ref_count = 1 → 0 → deleted
// Both destructors called! No leak!
```

#### Observer Pattern

```cpp
class Subject;

class Observer {
public:
    void notify(std::shared_ptr<Subject> subject) {
        std::cout << "Notified\n";
    }
};

class Subject {
    std::vector<std::weak_ptr<Observer>> observers;

public:
    void attach(std::shared_ptr<Observer> obs) {
        observers.push_back(obs);
    }

    void notify_all() {
        for (auto& weak_obs : observers) {
            if (std::shared_ptr<Observer> obs = weak_obs.lock()) {
                obs->notify(shared_from_this());
            }
        }
    }
};

// If observer is deleted, weak_ptr.lock() returns nullptr
// No dangling pointers!
```

#### Cache with Weak References

```cpp
class ImageCache {
    std::map<std::string, std::weak_ptr<Image>> cache;

public:
    std::shared_ptr<Image> load(const std::string& filename) {
        // Check cache
        auto it = cache.find(filename);
        if (it != cache.end()) {
            if (std::shared_ptr<Image> img = it->second.lock()) {
                return img;  // Image still in memory
            }
        }

        // Load image
        std::shared_ptr<Image> img = std::make_shared<Image>(filename);
        cache[filename] = img;  // Store weak reference
        return img;
    }
};

// When all shared_ptrs to image are destroyed, image is deleted
// Cache automatically updated (weak_ptr expires)
```

#### Checking Expiration

```cpp
std::shared_ptr<int> sp = std::make_shared<int>(42);
std::weak_ptr<int> wp = sp;

std::cout << wp.expired() << "\n";  // false (object alive)
std::cout << wp.use_count() << "\n";  // 1

sp.reset();  // Delete object

std::cout << wp.expired() << "\n";  // true (object deleted)
std::cout << wp.use_count() << "\n";  // 0
```

### RAII Pattern

Resource Acquisition Is Initialization - tie resource lifetime to object lifetime.

#### Principle

```cpp
// RAII:
// 1. Acquire resource in constructor
// 2. Release resource in destructor
// 3. Resource lifetime tied to object lifetime

class FileHandle {
    FILE* file;

public:
    FileHandle(const char* filename, const char* mode)
        : file(fopen(filename, mode))
    {
        if (!file) {
            throw std::runtime_error("Failed to open file");
        }
    }

    ~FileHandle() {
        if (file) {
            fclose(file);
        }
    }

    // Prevent copying (file handle shouldn't be copied)
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    FILE* get() { return file; }
};

// Usage
void process_file() {
    FileHandle file("data.txt", "r");
    // Use file.get()...
    // Automatic cleanup even if exception thrown!
}
```

#### Lock Guard

```cpp
#include <mutex>

std::mutex mtx;
int shared_data = 0;

void bad_example() {
    mtx.lock();

    shared_data++;

    if (error_condition) {
        return;  // BUG: Forgot to unlock!
    }

    mtx.unlock();
}

void good_example() {
    std::lock_guard<std::mutex> lock(mtx);  // RAII

    shared_data++;

    if (error_condition) {
        return;  // Automatic unlock
    }

    // Automatic unlock
}
```

#### Resource Manager Examples

**Socket:**
```cpp
class SocketHandle {
    int sockfd;

public:
    SocketHandle(const char* host, int port) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) throw std::runtime_error("Socket creation failed");

        // Connect...
    }

    ~SocketHandle() {
        if (sockfd >= 0) {
            close(sockfd);
        }
    }

    int get() { return sockfd; }
};
```

**Database Connection:**
```cpp
class DatabaseConnection {
    Connection* conn;

public:
    DatabaseConnection(const char* connstr) {
        conn = db_connect(connstr);
        if (!conn) throw std::runtime_error("Connection failed");
    }

    ~DatabaseConnection() {
        if (conn) {
            db_disconnect(conn);
        }
    }

    Connection* get() { return conn; }
};
```

**Memory Buffer:**
```cpp
class Buffer {
    char* data;
    size_t size;

public:
    Buffer(size_t n) : size(n) {
        data = new char[n];
    }

    ~Buffer() {
        delete[] data;
    }

    char* get() { return data; }
    size_t length() { return size; }
};
```

#### Advantages

1. **Automatic cleanup**: Resources always released
2. **Exception-safe**: Cleanup happens even if exception thrown
3. **Clear ownership**: Resource lifetime tied to scope
4. **No manual cleanup**: Can't forget to free

#### Best Practices

```cpp
// 1. Acquire in constructor, release in destructor
// 2. Delete copy operations if resource shouldn't be copied
// 3. Use unique_ptr/shared_ptr for dynamic allocations
// 4. Custom deleters for non-memory resources

// Example: Combining RAII with smart pointers
class Resource {
public:
    Resource() { std::cout << "Acquired\n"; }
    ~Resource() { std::cout << "Released\n"; }
};

void function() {
    std::unique_ptr<Resource> res = std::make_unique<Resource>();
    // Use resource...
    // Automatic cleanup
}
```

---

## Language-Specific Memory Management

### Python

#### Memory Model

Python uses:
1. **Reference Counting**: Primary mechanism
2. **Cycle Detector**: For circular references
3. **Memory Pools**: For small objects

#### Reference Counting

```python
import sys

# Create object (ref_count = 1)
a = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (1 + temporary reference from getrefcount)

# Add reference
b = a
print(sys.getrefcount(a))  # 3

# Remove reference
del b
print(sys.getrefcount(a))  # 2

# Remove last reference → object deleted
del a
```

#### Memory Management with `__del__`

```python
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Acquiring {name}")

    def __del__(self):
        print(f"Releasing {name}")

# Create object
r = Resource("File")  # "Acquiring File"

# Delete object
del r  # "Releasing File" (if no other references)

# Warning: __del__ timing is unpredictable with cycles
```

#### Garbage Collection

```python
import gc

# Get garbage collector stats
print(gc.get_count())  # (threshold0, threshold1, threshold2)

# Manual collection
gc.collect()  # Force collection, returns # of objects collected

# Disable/enable automatic collection
gc.disable()
# ... do work ...
gc.enable()

# Find uncollectable objects (usually due to __del__ in cycles)
gc.set_debug(gc.DEBUG_SAVEALL)
gc.collect()
print(gc.garbage)  # List of uncollectable objects
```

#### Circular Reference Example

```python
class Node:
    def __init__(self):
        self.ref = None

# Create cycle
node1 = Node()
node2 = Node()
node1.ref = node2
node2.ref = node1

# Delete external references
del node1
del node2

# Objects not immediately freed (circular reference)
# Cycle detector will eventually collect them

gc.collect()  # Force collection
```

#### Memory Optimization

**1. `__slots__`** (reduce memory overhead):
```python
# Without __slots__: each instance has a __dict__
class NormalClass:
    def __init__(self, x, y):
        self.x = x
        self.y = y

import sys
obj = NormalClass(1, 2)
print(sys.getsizeof(obj))  # e.g., 56 bytes

# With __slots__: no __dict__, fixed attributes
class OptimizedClass:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

obj2 = OptimizedClass(1, 2)
print(sys.getsizeof(obj2))  # e.g., 48 bytes

# obj2.z = 3  # AttributeError: no __dict__!
```

**Memory savings for many objects:**
```python
import sys

# 1 million objects without __slots__
objects1 = [NormalClass(i, i*2) for i in range(1000000)]
size1 = sum(sys.getsizeof(obj) for obj in objects1)

# 1 million objects with __slots__
objects2 = [OptimizedClass(i, i*2) for i in range(1000000)]
size2 = sum(sys.getsizeof(obj) for obj in objects2)

print(f"Normal: {size1/1024/1024:.2f} MB")
print(f"Optimized: {size2/1024/1024:.2f} MB")
print(f"Savings: {(1 - size2/size1)*100:.1f}%")

# Typical result: 30-50% memory savings
```

**2. Interning** (reuse immutable objects):
```python
# Small integers (-5 to 256) are interned
a = 100
b = 100
print(a is b)  # True (same object)

a = 1000
b = 1000
print(a is b)  # False (different objects)

# String interning
s1 = "hello"
s2 = "hello"
print(s1 is s2)  # True (interned)

# Force interning
import sys
s3 = sys.intern("unique_string")
s4 = sys.intern("unique_string")
print(s3 is s4)  # True
```

#### Memory Profiling

```python
import tracemalloc

# Start tracing
tracemalloc.start()

# Allocate memory
data = [i for i in range(1000000)]

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")

# Get top memory allocations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:5]:
    print(stat)

tracemalloc.stop()
```

### JavaScript

#### V8 Memory Management

JavaScript (V8 engine) uses generational garbage collection.

**Heap Structure:**
```
New Space (Young Generation):
  - New objects allocated here
  - Small (1-8 MB)
  - Fast, frequent GC (Scavenge)

Old Space (Old Generation):
  - Objects that survived multiple GCs
  - Larger (hundreds of MB)
  - Slower, less frequent GC (Mark-Sweep-Compact)

Large Object Space:
  - Objects > ~512 KB
  - Never moved
```

#### Memory Leaks in JavaScript

**Leak 1: Global Variables**
```javascript
// BAD: Creates global variable
function leak() {
    leakyVar = new Array(1000000);  // No var/let/const!
}

// GOOD: Use const/let
function noLeak() {
    const localVar = new Array(1000000);
}
```

**Leak 2: Event Listeners**
```javascript
// BAD: Event listener prevents GC
function setupElement() {
    const bigData = new Array(1000000);
    const element = document.getElementById('button');

    element.addEventListener('click', function() {
        console.log(bigData.length);  // Closes over bigData
    });
}

// GOOD: Remove listener when done
function setupElementCorrectly() {
    const bigData = new Array(1000000);
    const element = document.getElementById('button');

    const handler = function() {
        console.log(bigData.length);
    };

    element.addEventListener('click', handler);

    // Later:
    element.removeEventListener('click', handler);
}

// BETTER: Use AbortController
function setupElementBest() {
    const bigData = new Array(1000000);
    const element = document.getElementById('button');
    const controller = new AbortController();

    element.addEventListener('click', function() {
        console.log(bigData.length);
    }, { signal: controller.signal });

    // Later:
    controller.abort();  // Removes all listeners
}
```

**Leak 3: Timers**
```javascript
// BAD: setInterval keeps running
function startTimer() {
    const bigData = new Array(1000000);

    setInterval(() => {
        console.log(bigData.length);
    }, 1000);
}

// GOOD: Clear timer
function startTimerCorrectly() {
    const bigData = new Array(1000000);

    const timer = setInterval(() => {
        console.log(bigData.length);
    }, 1000);

    // Later:
    clearInterval(timer);
}
```

**Leak 4: Closures**
```javascript
// BAD: Closures retain entire scope
function createClosure() {
    const bigData = new Array(1000000);
    const smallData = [1, 2, 3];

    return function() {
        return smallData.length;  // Only uses smallData
    };
    // But bigData is still retained!
}

// GOOD: Minimize closure scope
function createClosureCorrectly() {
    const smallData = [1, 2, 3];

    return function() {
        return smallData.length;
    };
    // bigData not in closure scope
}
```

#### WeakMap and WeakRef

**WeakMap** (weak references to keys):
```javascript
const cache = new WeakMap();

let obj = { data: 'value' };
cache.set(obj, 'cached data');

console.log(cache.get(obj));  // 'cached data'

obj = null;  // Object can be GC'd
// cache entry automatically removed
```

**WeakRef** (ES2021):
```javascript
let obj = { data: 'value' };
const weakRef = new WeakRef(obj);

console.log(weakRef.deref());  // { data: 'value' }

obj = null;  // Object can be GC'd

// Later:
console.log(weakRef.deref());  // undefined (if GC'd)
```

#### Memory Profiling (Chrome DevTools)

```javascript
// 1. Take heap snapshot
// DevTools → Memory → Take snapshot

// 2. Compare snapshots
// Take snapshot before
const leak = [];
function allocate() {
    leak.push(new Array(1000000));
}

allocate();
// Take snapshot after

// 3. Allocation timeline
// DevTools → Memory → Allocation instrumentation on timeline

// 4. Force GC
// DevTools → Performance → Collect garbage
```

### Go

#### Garbage Collector

Go uses concurrent mark-sweep GC with tri-color marking.

**Characteristics:**
- Concurrent: Runs alongside application
- Low latency: Pause times < 1 ms (typically)
- Non-generational: Single heap (no young/old split)

#### Memory Allocation

```go
// Stack allocation (automatic)
func stackAlloc() {
    x := 42              // On stack
    arr := [10]int{}     // On stack
}

// Heap allocation (escapes to heap)
func heapAlloc() *int {
    x := 42
    return &x  // Escapes to heap
}

// Slice (heap allocation)
func sliceAlloc() {
    s := make([]int, 1000)  // On heap
    _ = s
}
```

**Escape Analysis:**
```go
// Check what escapes to heap
// go build -gcflags='-m'

func example() {
    x := 42         // stack
    y := &x         // x escapes to heap (address taken and returned)
    _ = y
}
```

#### Manual GC Control

```go
import "runtime"

func main() {
    // Force GC
    runtime.GC()

    // Set GC percentage (default: 100)
    // GOGC=50: GC when heap grows 50%
    // GOGC=200: GC when heap triples
    runtime.SetGCPercent(200)

    // Get memory stats
    var m runtime.MemStats
    runtime.ReadMemStats(&m)

    fmt.Printf("Alloc: %d MB\n", m.Alloc / 1024 / 1024)
    fmt.Printf("TotalAlloc: %d MB\n", m.TotalAlloc / 1024 / 1024)
    fmt.Printf("Sys: %d MB\n", m.Sys / 1024 / 1024)
    fmt.Printf("NumGC: %d\n", m.NumGC)
}
```

#### Memory Optimization

**1. Sync.Pool** (object reuse):
```go
var bufferPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func processData(data []byte) {
    // Get buffer from pool
    buf := bufferPool.Get().(*bytes.Buffer)
    buf.Reset()

    // Use buffer
    buf.Write(data)
    processBuffer(buf)

    // Return to pool
    bufferPool.Put(buf)
}
```

**2. Avoid allocations**:
```go
// BAD: Allocates on every call
func bad(n int) []int {
    return make([]int, n)
}

// GOOD: Reuse buffer
type Processor struct {
    buffer []int
}

func (p *Processor) process(n int) []int {
    if cap(p.buffer) < n {
        p.buffer = make([]int, n)
    }
    return p.buffer[:n]
}
```

**3. Preallocate slices**:
```go
// BAD: Many reallocations
func bad() []int {
    var result []int
    for i := 0; i < 1000000; i++ {
        result = append(result, i)  // Reallocates many times
    }
    return result
}

// GOOD: Preallocate
func good() []int {
    result := make([]int, 0, 1000000)
    for i := 0; i < 1000000; i++ {
        result = append(result, i)  // No reallocations
    }
    return result
}
```

### Rust

#### Ownership System

Rust uses compile-time ownership tracking instead of garbage collection.

**Rules:**
1. Each value has a single owner
2. When owner goes out of scope, value is dropped
3. Only one mutable reference OR multiple immutable references

```rust
fn main() {
    let s = String::from("hello");  // s owns the string

    takes_ownership(s);  // s moved, no longer valid

    // println!("{}", s);  // ERROR: s was moved
}

fn takes_ownership(s: String) {
    println!("{}", s);
}  // s dropped here
```

#### Borrowing

```rust
fn main() {
    let s = String::from("hello");

    // Immutable borrow
    let len = calculate_length(&s);  // Borrow, don't move

    println!("Length of '{}' is {}", s, len);  // s still valid
}

fn calculate_length(s: &String) -> usize {
    s.len()
}  // s goes out of scope, but doesn't drop (just a reference)
```

**Mutable borrows:**
```rust
fn main() {
    let mut s = String::from("hello");

    change(&mut s);

    println!("{}", s);  // "hello, world"
}

fn change(s: &mut String) {
    s.push_str(", world");
}
```

**Borrow rules enforced at compile time:**
```rust
fn main() {
    let mut s = String::from("hello");

    let r1 = &s;
    let r2 = &s;  // OK: multiple immutable borrows
    // let r3 = &mut s;  // ERROR: can't borrow as mutable while immutable borrows exist

    println!("{} {}", r1, r2);
}
```

#### Lifetimes

```rust
// Lifetime annotations
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string");
    let string2 = String::from("short");

    let result = longest(&string1, &string2);
    println!("Longest: {}", result);
}

// Compiler ensures returned reference doesn't outlive inputs
```

#### Smart Pointers

**Box** (heap allocation):
```rust
fn main() {
    let b = Box::new(5);  // Allocate on heap
    println!("b = {}", b);
}  // b dropped, heap memory freed
```

**Rc** (reference counting):
```rust
use std::rc::Rc;

fn main() {
    let a = Rc::new(5);           // ref_count = 1
    let b = Rc::clone(&a);        // ref_count = 2
    let c = Rc::clone(&a);        // ref_count = 3

    println!("count: {}", Rc::strong_count(&a));  // 3
}  // All dropped, memory freed when count reaches 0
```

**RefCell** (interior mutability):
```rust
use std::cell::RefCell;

fn main() {
    let value = RefCell::new(5);

    *value.borrow_mut() = 10;  // Runtime borrow checking

    println!("{}", value.borrow());
}
```

#### Zero-Cost Abstractions

```rust
// No runtime overhead!
fn main() {
    let v = vec![1, 2, 3];

    // Iterator: compiled to same code as manual loop
    let sum: i32 = v.iter().map(|x| x * 2).sum();

    println!("{}", sum);
}

// Equivalent to:
fn manual() {
    let v = vec![1, 2, 3];
    let mut sum = 0;
    for x in &v {
        sum += x * 2;
    }
    println!("{}", sum);
}

// Both compile to identical assembly!
```

### Java

#### Heap Structure

```
Heap:
+---------------------------+
| Young Generation          |
|  - Eden Space             |
|  - Survivor Space 0       |
|  - Survivor Space 1       |
+---------------------------+
| Old Generation (Tenured)  |
+---------------------------+
| Metaspace (JDK 8+)        |
| (Class metadata)          |
+---------------------------+
```

#### Object Lifecycle

```java
public class ObjectLifecycle {
    public static void main(String[] args) {
        // 1. Allocation in Eden space
        MyObject obj = new MyObject();  // Allocated in Eden

        // 2. Minor GC moves survivors to Survivor space
        // (happens automatically when Eden fills)

        // 3. After several GCs, promoted to Old Generation

        // 4. When obj = null, object becomes eligible for GC
        obj = null;

        // 5. Major GC (when old gen fills) reclaims object
    }
}
```

#### Garbage Collectors

**1. Serial GC** (single-threaded):
```bash
java -XX:+UseSerialGC MyApp
# Good for: Small heaps, single-CPU systems
```

**2. Parallel GC** (multi-threaded):
```bash
java -XX:+UseParallelGC MyApp
# Good for: High throughput, batch processing
```

**3. CMS (Concurrent Mark Sweep)**:
```bash
java -XX:+UseConcMarkSweepGC MyApp
# Good for: Low latency (deprecated in JDK 9)
```

**4. G1 (Garbage First)**:
```bash
java -XX:+UseG1GC -XX:MaxGCPauseMillis=200 MyApp
# Good for: Balanced throughput/latency, large heaps
```

**5. ZGC** (ultra-low latency):
```bash
java -XX:+UseZGC MyApp
# Good for: Sub-10ms pauses, very large heaps (TB+)
```

**6. Shenandoah**:
```bash
java -XX:+UseShenandoahGC MyApp
# Good for: Low latency, concurrent compaction
```

#### Memory Tuning

```bash
# Heap size
java -Xms2g -Xmx4g MyApp  # Initial 2GB, max 4GB

# Young generation size
java -Xmn1g MyApp  # 1GB young gen

# Metaspace size (class metadata)
java -XX:MetaspaceSize=256m -XX:MaxMetaspaceSize=512m MyApp

# GC logging
java -Xlog:gc*:file=gc.log:time,uptime:filecount=5,filesize=100m MyApp
```

#### WeakReference, SoftReference, PhantomReference

```java
import java.lang.ref.*;

public class References {
    public static void main(String[] args) {
        Object obj = new Object();

        // Strong reference: never GC'd while reachable
        Object strong = obj;

        // Weak reference: GC'd even if memory available
        WeakReference<Object> weak = new WeakReference<>(obj);
        System.out.println(weak.get());  // Returns object
        obj = null;
        System.gc();
        System.out.println(weak.get());  // null (GC'd)

        // Soft reference: GC'd only when memory low
        obj = new Object();
        SoftReference<Object> soft = new SoftReference<>(obj);
        obj = null;
        // soft.get() returns object until memory pressure

        // Phantom reference: for cleanup actions
        obj = new Object();
        ReferenceQueue<Object> queue = new ReferenceQueue<>();
        PhantomReference<Object> phantom = new PhantomReference<>(obj, queue);
        obj = null;
        System.gc();
        // phantom.get() always returns null
        // Used for post-finalization cleanup
    }
}
```

---

## Memory Profiling

### Profiling Tools

#### Valgrind (Linux)

**Memcheck** (memory error detector):
```bash
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./myapp
```

Detects:
- Memory leaks
- Use-after-free
- Double-free
- Invalid reads/writes
- Uninitialized memory usage

**Massif** (heap profiler):
```bash
valgrind --tool=massif ./myapp
ms_print massif.out.<pid>
```

Output:
```
KB
1.000^
     |
     |
     |                           @@@@@@@@
     |                      @@@@@        @@@@@
     |                 @@@@@                  @@@@@
     |            @@@@@                            @@@@@
     |       @@@@@                                      @@@@@
     |  @@@@@                                                @@@@@
0    +----------------------------------------------------------------------->
     0                                                                   100 s
```

**Cachegrind** (cache profiler):
```bash
valgrind --tool=cachegrind ./myapp
cg_annotate cachegrind.out.<pid>
```

#### Heaptrack (Linux)

Modern heap profiler with GUI:
```bash
heaptrack ./myapp
heaptrack_gui heaptrack.myapp.<pid>.gz
```

Shows:
- Allocation flamegraphs
- Memory timeline
- Top allocators
- Leak detection

#### Instruments (macOS)

Xcode profiling tool:
```bash
# Launch Instruments
instruments -t Leaks ./myapp

# Or from Xcode: Product → Profile (⌘I)
```

Templates:
- **Leaks**: Detect memory leaks
- **Allocations**: Track all allocations
- **VM Tracker**: Virtual memory usage

#### Windows Memory Diagnostic

**Visual Studio Diagnostic Tools:**
- Debug → Windows → Show Diagnostic Tools
- Shows memory usage timeline
- Snapshot heap for analysis

**Performance Profiler:**
- Debug → Performance Profiler
- Select ".NET Object Allocation Tracking"
- Analyze allocation flamegraphs

### Memory Leak Detection Tools

#### LeakSanitizer

```bash
# Standalone
gcc -fsanitize=leak -g -o myapp myapp.c
./myapp

# Or included with AddressSanitizer
gcc -fsanitize=address -g -o myapp myapp.c
./myapp
```

Example output:
```
=================================================================
==12345==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 100 byte(s) in 1 object(s) allocated from:
    #0 0x7f8b7cc63537 in malloc
    #1 0x400b95 in main myapp.c:10

SUMMARY: LeakSanitizer: 100 byte(s) leaked in 1 allocation(s).
```

#### mtrace (glibc)

GNU C library's malloc tracer:

```c
#include <mcheck.h>

int main() {
    mtrace();  // Start tracing

    char* leak = malloc(100);
    // Forgot to free!

    muntrace();  // Stop tracing
    return 0;
}
```

```bash
gcc -g -o myapp myapp.c
export MALLOC_TRACE=mtrace.log
./myapp
mtrace myapp mtrace.log
```

Output:
```
Memory not freed:
-----------------
   Address     Size     Caller
0x55e4d789ef00   0x64  at /path/to/myapp.c:10
```

#### Python memory_profiler

```python
from memory_profiler import profile

@profile
def my_function():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

if __name__ == '__main__':
    my_function()
```

```bash
python -m memory_profiler myapp.py
```

Output:
```
Line #    Mem usage    Increment  Line Contents
================================================
     3   38.816 MiB   38.816 MiB  @profile
     4                             def my_function():
     5   46.492 MiB    7.676 MiB      a = [1] * (10 ** 6)
     6  199.344 MiB  152.852 MiB      b = [2] * (2 * 10 ** 7)
     7   46.492 MiB -152.852 MiB      del b
     8   46.492 MiB    0.000 MiB      return a
```

### Heap Profiling

#### jemalloc Profiling

```bash
# Compile with jemalloc
gcc -o myapp myapp.c -ljemalloc

# Enable profiling
export MALLOC_CONF=prof:true,prof_prefix:jeprof.out
./myapp

# Analyze profile
jeprof --pdf myapp jeprof.out.<pid>.heap > profile.pdf
```

#### gperftools (Google Performance Tools)

```c
#include <gperftools/heap-profiler.h>

int main() {
    HeapProfilerStart("myapp");

    // Your code here
    for (int i = 0; i < 1000000; i++) {
        char* ptr = malloc(100);
        // ...
    }

    HeapProfilerStop();
    return 0;
}
```

```bash
gcc -o myapp myapp.c -ltcmalloc
./myapp
pprof --pdf myapp myapp.0001.heap > heap_profile.pdf
```

#### Java Flight Recorder (JFR)

```bash
# Start recording
java -XX:StartFlightRecording=duration=60s,filename=recording.jfr MyApp

# Or attach to running process
jcmd <pid> JFR.start duration=60s filename=recording.jfr

# Analyze with JDK Mission Control
jmc recording.jfr
```

#### Chrome DevTools (JavaScript)

```javascript
// Heap snapshot
// DevTools → Memory → Take snapshot

// Example: Find detached DOM nodes
function createLeak() {
    const div = document.createElement('div');
    div.innerHTML = '<p>Content</p>';

    window.leakedNode = div;  // Prevents GC
}

// 1. Take snapshot
// 2. Run createLeak()
// 3. Take snapshot
// 4. Compare snapshots → find "Detached DOM tree"
```

---

## Performance Optimization

### Cache-Friendly Data Structures

#### Cache Hierarchy

```
CPU Registers: ~1 cycle (~0.3 ns)
L1 Cache: ~4 cycles (~1 ns), 32-64 KB per core
L2 Cache: ~12 cycles (~3 ns), 256-512 KB per core
L3 Cache: ~40 cycles (~10 ns), 8-64 MB shared
RAM: ~200 cycles (~60 ns), GB+
```

#### Cache Lines

Modern CPUs fetch memory in cache lines (typically 64 bytes).

```c
// BAD: False sharing
struct {
    int counter1;  // Offset 0
    int counter2;  // Offset 4
} shared;

// Thread 1
shared.counter1++;  // Invalidates entire cache line

// Thread 2
shared.counter2++;  // Must reload cache line (slow!)
```

```c
// GOOD: Padding to separate cache lines
struct {
    int counter1;
    char padding[60];  // Pad to 64 bytes
    int counter2;
} shared;

// Thread 1 and 2 now use different cache lines
```

#### Array of Structures vs Structure of Arrays

**Array of Structures (AoS):**
```c
struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float mass;
};

struct Particle particles[1000];

// Update positions
for (int i = 0; i < 1000; i++) {
    particles[i].x += particles[i].vx;
    particles[i].y += particles[i].vy;
    particles[i].z += particles[i].vz;
}

// Cache-unfriendly: loads entire struct, wastes bandwidth
```

**Structure of Arrays (SoA):**
```c
struct Particles {
    float x[1000];
    float y[1000];
    float z[1000];
    float vx[1000];
    float vy[1000];
    float vz[1000];
    float mass[1000];
};

struct Particles particles;

// Update positions
for (int i = 0; i < 1000; i++) {
    particles.x[i] += particles.vx[i];
    particles.y[i] += particles.vy[i];
    particles.z[i] += particles.vz[i];
}

// Cache-friendly: sequential access, full cache line utilization
```

**Performance Comparison:**
```
AoS: ~100 ms (many cache misses)
SoA: ~20 ms (few cache misses)
Speedup: 5x!
```

#### Prefetching

```c
// Manual prefetching
#include <xmmintrin.h>

void process_array(int* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // Prefetch next iteration
        if (i + 8 < n) {
            _mm_prefetch(&arr[i + 8], _MM_HINT_T0);
        }

        // Process current
        arr[i] = arr[i] * 2 + 1;
    }
}
```

### Memory Access Patterns

#### Sequential vs Random Access

```c
#define SIZE (1024 * 1024 * 100)  // 100M ints

int* arr = malloc(SIZE * sizeof(int));

// Sequential access (cache-friendly)
clock_t start = clock();
for (int i = 0; i < SIZE; i++) {
    arr[i] = i;
}
double seq_time = (double)(clock() - start) / CLOCKS_PER_SEC;

// Random access (cache-unfriendly)
start = clock();
for (int i = 0; i < SIZE; i++) {
    int index = rand() % SIZE;
    arr[index] = i;
}
double rand_time = (double)(clock() - start) / CLOCKS_PER_SEC;

printf("Sequential: %.3f s\n", seq_time);
printf("Random: %.3f s\n", rand_time);
printf("Ratio: %.2fx\n", rand_time / seq_time);

// Typical result: Random is 10-50x slower!
```

#### Loop Tiling (Blocking)

Improve cache locality by processing data in blocks:

```c
// Matrix multiplication: Naive (cache-unfriendly)
void matmul_naive(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
                // B accessed with stride N (cache miss!)
            }
            C[i*N + j] = sum;
        }
    }
}

// Matrix multiplication: Tiled (cache-friendly)
#define BLOCK_SIZE 32

void matmul_tiled(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // Process BLOCK_SIZE x BLOCK_SIZE sub-matrix
                for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj++) {
                        double sum = C[ii*N + jj];
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk++) {
                            sum += A[ii*N + kk] * B[kk*N + jj];
                        }
                        C[ii*N + jj] = sum;
                    }
                }
            }
        }
    }
}

// Performance (N=1024):
// Naive: 10.5 seconds
// Tiled: 1.2 seconds
// Speedup: 8.75x!
```

### Copy-on-Write

Share memory until modification, then copy.

#### Fork Example (Unix)

```c
#include <unistd.h>
#include <sys/wait.h>

int main() {
    char* data = malloc(1000000000);  // 1 GB
    memset(data, 0, 1000000000);

    pid_t pid = fork();

    if (pid == 0) {
        // Child process
        // Shares parent's memory (COW)
        sleep(1);

        // Write triggers COW (copy page)
        data[0] = 42;

        exit(0);
    } else {
        // Parent process
        wait(NULL);
    }

    free(data);
    return 0;
}

// fork() is instant (doesn't copy 1 GB)
// Only modified pages are copied
```

#### String Implementation

```cpp
class CowString {
    struct Data {
        char* str;
        size_t len;
        std::atomic<int> ref_count;
    };

    Data* data;

    void detach() {
        if (data->ref_count > 1) {
            // Copy string (COW)
            Data* new_data = new Data{
                new char[data->len + 1],
                data->len,
                1
            };
            memcpy(new_data->str, data->str, data->len + 1);

            data->ref_count--;
            data = new_data;
        }
    }

public:
    CowString(const char* s) {
        data = new Data{
            new char[strlen(s) + 1],
            strlen(s),
            1
        };
        strcpy(data->str, s);
    }

    // Copy constructor (shares data)
    CowString(const CowString& other) : data(other.data) {
        data->ref_count++;
    }

    // Modify: triggers COW
    void set_char(size_t i, char c) {
        detach();  // Copy if shared
        data->str[i] = c;
    }

    // Read: no COW
    char get_char(size_t i) const {
        return data->str[i];
    }

    ~CowString() {
        if (--data->ref_count == 0) {
            delete[] data->str;
            delete data;
        }
    }
};
```

### Memory-Mapped Files

Map files directly into virtual memory.

#### Basic Usage (POSIX)

```c
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    // Open file
    int fd = open("data.bin", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Get file size
    struct stat sb;
    fstat(fd, &sb);
    size_t size = sb.st_size;

    // Memory-map file
    char* data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    // Access file as memory
    data[0] = 'H';
    data[1] = 'i';

    // Changes written back to file (eventually)
    msync(data, size, MS_SYNC);  // Force write

    // Unmap
    munmap(data, size);
    close(fd);

    return 0;
}
```

#### Advantages

1. **Lazy loading**: Pages loaded on demand
2. **Shared memory**: Multiple processes can map same file
3. **No explicit I/O**: OS handles reads/writes
4. **Large files**: Don't need to fit in RAM

#### Example: Large File Processing

```c
void process_large_file(const char* filename) {
    int fd = open(filename, O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);

    size_t size = sb.st_size;
    char* data = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    // Process file in chunks
    size_t chunk_size = 1024 * 1024;  // 1 MB
    for (size_t offset = 0; offset < size; offset += chunk_size) {
        size_t len = (offset + chunk_size < size) ? chunk_size : (size - offset);
        process_chunk(data + offset, len);
    }

    munmap(data, size);
    close(fd);
}

// OS loads only accessed pages (efficient!)
```

#### Example: Shared Memory IPC

```c
// Process 1: Create shared memory
int fd = shm_open("/my_shm", O_CREAT | O_RDWR, 0666);
ftruncate(fd, 4096);

int* shared = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
*shared = 42;

// Process 2: Attach to shared memory
int fd = shm_open("/my_shm", O_RDWR, 0666);
int* shared = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

printf("Value: %d\n", *shared);  // 42
```

---

## Common Pitfalls and Best Practices

### Common Pitfalls

**1. Memory Leaks**
```c
// BAD
char* get_string() {
    char* str = malloc(100);
    strcpy(str, "Hello");
    return str;  // Caller must remember to free!
}

// GOOD: Document ownership
char* get_string() {
    // Caller owns returned pointer and must free it
    char* str = malloc(100);
    strcpy(str, "Hello");
    return str;
}

// BETTER: Use output parameter
void get_string(char* buffer, size_t size) {
    strncpy(buffer, "Hello", size - 1);
    buffer[size - 1] = '\0';
}
```

**2. Dangling Pointers**
```c
// BAD
int* get_local() {
    int x = 42;
    return &x;  // Returns address of stack variable!
}

// GOOD
int* get_heap() {
    int* x = malloc(sizeof(int));
    *x = 42;
    return x;
}
```

**3. Buffer Overflows**
```c
// BAD
char buffer[10];
strcpy(buffer, user_input);  // What if user_input is longer?

// GOOD
char buffer[10];
strncpy(buffer, user_input, sizeof(buffer) - 1);
buffer[sizeof(buffer) - 1] = '\0';

// BETTER (C11)
strncpy_s(buffer, sizeof(buffer), user_input, _TRUNCATE);
```

**4. Uninitialized Memory**
```c
// BAD
int* arr = malloc(10 * sizeof(int));
printf("%d\n", arr[0]);  // Undefined value!

// GOOD
int* arr = calloc(10, sizeof(int));  // Zero-initialized
printf("%d\n", arr[0]);  // 0
```

**5. Memory Alignment Issues**
```c
// BAD (may crash on some architectures)
char buffer[100];
int* ptr = (int*)&buffer[1];  // Unaligned!
*ptr = 42;  // May crash or be slow

// GOOD
int* ptr = (int*)&buffer[0];  // Aligned to int boundary
*ptr = 42;
```

### Best Practices

**1. Ownership Clarity**
```cpp
// Clear ownership with unique_ptr
std::unique_ptr<Resource> create_resource() {
    return std::make_unique<Resource>();
}

void use_resource() {
    auto res = create_resource();  // Ownership transferred
    // Use res...
    // Automatic cleanup
}
```

**2. RAII Pattern**
```cpp
class FileHandle {
    FILE* file;
public:
    FileHandle(const char* name, const char* mode)
        : file(fopen(name, mode))
    {
        if (!file) throw std::runtime_error("Failed to open file");
    }

    ~FileHandle() {
        if (file) fclose(file);
    }

    FILE* get() { return file; }
};

// Usage
void process_file() {
    FileHandle file("data.txt", "r");
    // Use file.get()...
    // Automatic cleanup even if exception thrown
}
```

**3. Bounds Checking**
```c
void safe_copy(char* dest, size_t dest_size, const char* src) {
    if (strlen(src) >= dest_size) {
        // Handle error
        return;
    }
    strcpy(dest, src);
}
```

**4. Null Pointer Checks**
```c
void process(int* ptr) {
    if (!ptr) {
        // Handle null pointer
        return;
    }

    *ptr = 42;
}
```

**5. Use Static Analysis**
```bash
# Clang static analyzer
scan-build gcc -o myapp myapp.c

# Cppcheck
cppcheck --enable=all myapp.c

# Valgrind
valgrind --leak-check=full ./myapp
```

**6. Memory Profiling in Development**
```bash
# Compile with sanitizers during development
gcc -fsanitize=address -fsanitize=undefined -g -o myapp myapp.c

# Run tests
./myapp
```

**7. Documentation**
```c
/**
 * Creates a new string.
 * @return Newly allocated string. Caller must free with free().
 */
char* create_string(const char* src);

/**
 * Processes data in buffer.
 * @param buffer Buffer to process (not owned, not modified).
 */
void process_data(const char* buffer);

/**
 * Takes ownership of resource.
 * @param resource Resource to take ownership of. Will be freed.
 */
void take_resource(Resource* resource);
```

**8. Defensive Programming**
```c
void safe_free(void** ptr) {
    if (ptr && *ptr) {
        free(*ptr);
        *ptr = NULL;
    }
}

// Usage
char* str = malloc(100);
safe_free((void**)&str);
safe_free((void**)&str);  // Safe to call twice
```

**9. Memory Budgets**
```c
#define MAX_MEMORY_MB 100

static size_t allocated_memory = 0;

void* tracked_malloc(size_t size) {
    if (allocated_memory + size > MAX_MEMORY_MB * 1024 * 1024) {
        fprintf(stderr, "Memory budget exceeded\n");
        return NULL;
    }

    void* ptr = malloc(size);
    if (ptr) {
        allocated_memory += size;
    }
    return ptr;
}

void tracked_free(void* ptr, size_t size) {
    if (ptr) {
        free(ptr);
        allocated_memory -= size;
    }
}
```

**10. Testing for Leaks**
```bash
#!/bin/bash
# run_tests.sh

# Compile with sanitizers
gcc -fsanitize=address -g -o test test.c

# Run tests
./test

# Check exit code
if [ $? -ne 0 ]; then
    echo "Tests failed or memory errors detected"
    exit 1
fi

echo "All tests passed"
```

---

## Summary

Memory management is a fundamental aspect of systems programming. Key takeaways:

1. **Understand your memory model**: Stack, heap, static allocation
2. **Choose appropriate strategies**: Manual, GC, smart pointers, arenas
3. **Profile before optimizing**: Measure, don't guess
4. **Use tools**: Valgrind, AddressSanitizer, profilers
5. **Follow best practices**: RAII, ownership clarity, bounds checking
6. **Test thoroughly**: Static analysis, dynamic analysis, leak detection

Different languages and use cases require different approaches:
- **C/C++**: Manual management or smart pointers
- **Python/Java/Go**: Garbage collection
- **Rust**: Compile-time ownership
- **Games/Real-time**: Arenas, pools, manual control

The right choice depends on your performance requirements, development time constraints, and correctness guarantees needed.
