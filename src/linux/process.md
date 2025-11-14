# Linux Process Management

## Table of Contents
- [Overview](#overview)
- [Process Fundamentals](#process-fundamentals)
- [Process Identification](#process-identification)
- [Process Memory Layout](#process-memory-layout)
- [Stack Management](#stack-management)
- [Heap Management](#heap-management)
- [Shared Objects (SO)](#shared-objects-so)
- [Process Lifecycle](#process-lifecycle)
- [Process Operations](#process-operations)
- [Process States](#process-states)
- [Process Management Tools](#process-management-tools)
- [/proc Filesystem](#proc-filesystem)
- [Inter-Process Communication](#inter-process-communication)
- [Advanced Patterns](#advanced-patterns)
- [Practical Examples](#practical-examples)

---

## Overview

A **process** is an instance of a running program. It's the fundamental unit of execution in Unix/Linux systems. Each process has its own:
- Address space (memory)
- Process ID (PID)
- File descriptors
- Security attributes (UID, GID)
- Execution context (registers, PC, stack pointer)

### Process vs Thread

- **Process**: Independent execution unit with separate memory space
- **Thread**: Lightweight execution unit sharing the same memory space within a process
- Threads share: text, data, heap, file descriptors
- Threads have separate: stack, registers, thread-local storage

---

## Process Fundamentals

### What is a Process?

A process consists of:
1. **Program code** (text segment)
2. **Current activity** (program counter, register values)
3. **Stack** (temporary data: function parameters, return addresses, local variables)
4. **Data section** (global variables)
5. **Heap** (dynamically allocated memory)

### Process Attributes

```c
struct task_struct {
    pid_t pid;                    // Process ID
    pid_t tgid;                   // Thread group ID
    struct task_struct *parent;   // Parent process
    struct list_head children;    // List of child processes
    struct mm_struct *mm;         // Memory descriptor
    struct files_struct *files;   // Open file descriptors
    // ... hundreds more fields
};
```

---

## Process Identification

### PID (Process ID)

Every process has a unique Process ID:

```bash
# Get current process PID
echo $$

# Get PID of a command
pgrep firefox
pidof firefox

# Kill a process by PID
kill -9 12345
```

**Key PIDs:**
- `PID 0`: Scheduler (kernel space)
- `PID 1`: init/systemd (first user space process)
- `PID 2`: kthreadd (kernel thread daemon)

### PPID (Parent Process ID)

Every process (except PID 1) has a parent:

```bash
# View parent-child relationship
ps -ef | grep process_name
pstree -p

# Get PPID programmatically
cat /proc/$$/status | grep PPid
```

**Orphan Process**: When a parent dies before its child, the child is adopted by init (PID 1).

**Zombie Process**: Child process that has terminated but parent hasn't read its exit status via `wait()`.

### Process Group ID (PGID)

Processes can be grouped for signal management:

```bash
# Send signal to entire process group
kill -TERM -12345  # Negative PID = process group

# Get process group
ps -o pid,pgid,cmd

# Job control
./long_running_task &  # Background job
jobs                   # List jobs
fg %1                  # Foreground job
```

### Session ID (SID)

A session is a collection of process groups:

```bash
# Get session ID
ps -o pid,sid,cmd

# Create new session (daemon pattern)
setsid ./daemon_process
```

### User and Group IDs

Security context:

```bash
# Real, Effective, Saved UIDs
cat /proc/$$/status | grep -E "Uid|Gid"

# UID types:
# - Real UID (RUID): User who started the process
# - Effective UID (EUID): Used for permission checks
# - Saved UID (SUID): Previous EUID (for privilege dropping)
```

---

## Process Memory Layout

A Linux process has a well-defined virtual memory layout:

```
High Address (0xFFFFFFFF / 0x7FFFFFFFFFFF on 64-bit)
┌─────────────────────────────────────┐
│         Kernel Space                │  <- Not accessible from user space
│         (1GB / 128TB)               │
├─────────────────────────────────────┤ 0xC0000000 (32-bit) / 0x00007FFFFFFFFFFF (64-bit)
│                                     │
│         Stack                       │  <- Grows downward (high → low)
│         (Local variables,           │
│          function calls)            │
│              ↓                      │
│                                     │
│         ...                         │
│                                     │
│              ↑                      │
│         Memory Mapping              │  <- mmap(), shared libraries
│         (Shared objects, mmap)      │
│              ↑                      │
│                                     │
│         ...                         │
│                                     │
│              ↑                      │
│         Heap                        │  <- Grows upward (low → high)
│         (Dynamic memory: malloc)    │
│                                     │
├─────────────────────────────────────┤
│         BSS Segment                 │  <- Uninitialized global/static vars
│         (Uninitialized data)        │     Initialized to 0
├─────────────────────────────────────┤
│         Data Segment                │  <- Initialized global/static vars
│         (Initialized data)          │
├─────────────────────────────────────┤
│         Text Segment                │  <- Program code (read-only)
│         (Code)                      │
└─────────────────────────────────────┘ Low Address (0x00000000)
```

### Segments Explained

#### 1. Text Segment (Code)
- Contains executable instructions
- Read-only and shareable
- Multiple processes can share the same text segment

```bash
# View segments
readelf -l /bin/ls
objdump -h /bin/ls

# Check if text is read-only
cat /proc/$$/maps | grep r-xp
```

#### 2. Data Segment
- Initialized global and static variables
- Read-write
- Size known at compile time

```c
int global_var = 42;              // Data segment
static int static_var = 100;      // Data segment
const int const_var = 200;        // May be in read-only data or text

int main() {
    // ...
}
```

#### 3. BSS Segment (Block Started by Symbol)
- Uninitialized global and static variables
- Automatically initialized to 0
- Doesn't occupy space in executable file

```c
int global_uninit;                // BSS
static int static_uninit;         // BSS

int main() {
    printf("%d\n", global_uninit); // Prints 0
}
```

**Why BSS?** Saves disk space. Instead of storing zeros in the executable, the loader allocates and zeros the memory at runtime.

#### 4. Heap Segment
- Dynamically allocated memory
- Grows upward (toward higher addresses)
- Managed by `malloc()`, `calloc()`, `realloc()`, `free()`
- See [Heap Management](#heap-management) for details

#### 5. Memory Mapping Segment
- Shared libraries (.so files)
- Memory-mapped files (`mmap()`)
- Anonymous mappings
- Position Independent Code (PIC)

#### 6. Stack Segment
- Function call frames
- Local variables
- Function parameters
- Return addresses
- Grows downward (toward lower addresses)
- See [Stack Management](#stack-management) for details

### Viewing Memory Layout

```bash
# View process memory map
cat /proc/$$/maps

# Example output:
# 00400000-00401000 r-xp ...  /bin/bash     <- Text
# 00600000-00601000 r--p ...  /bin/bash     <- Data
# 00601000-00602000 rw-p ...  /bin/bash     <- Data
# 01a15000-01a36000 rw-p ...  [heap]        <- Heap
# 7fff12345000-...  rw-p ...  [stack]       <- Stack
# 7f1234567000-...  r-xp ...  libc.so.6     <- Shared lib

# Memory usage
pmap -x $$
cat /proc/$$/status | grep -E "VmSize|VmRSS|VmData|VmStk"

# Detailed memory info
smem -p $$
```

---

## Stack Management

### Stack Fundamentals

The stack is a contiguous region of memory that:
- Stores local variables
- Manages function calls (call stack)
- Saves return addresses
- Passes function arguments
- Grows **downward** (high address → low address)

### Stack Frame

Each function call creates a **stack frame** (activation record):

```
High Address
┌──────────────────────┐
│   Previous frame     │
├──────────────────────┤ <- Previous Frame Pointer (FP)
│   Arguments          │
├──────────────────────┤
│   Return Address     │
├──────────────────────┤ <- Frame Pointer (FP/RBP)
│   Saved FP           │
├──────────────────────┤
│   Local Variables    │
├──────────────────────┤
│   Temporary Space    │
├──────────────────────┤ <- Stack Pointer (SP/RSP)
│   (Free space)       │
└──────────────────────┘
Low Address
```

### Stack Operations

```c
void func(int a, int b) {
    int x = 10;        // Local variable on stack
    int arr[100];      // Array on stack (400 bytes)

    // Stack frame contains:
    // - Parameters: a, b
    // - Return address
    // - Saved frame pointer
    // - Local vars: x, arr[100]
}

int main() {
    func(5, 7);
    return 0;
}
```

**Assembly view (x86-64 simplified):**

```asm
main:
    push rbp              ; Save old frame pointer
    mov rbp, rsp          ; Set new frame pointer
    mov edi, 5            ; First argument (a)
    mov esi, 7            ; Second argument (b)
    call func             ; Push return address and jump

func:
    push rbp              ; Save caller's frame pointer
    mov rbp, rsp          ; Set new frame pointer
    sub rsp, 416          ; Allocate space for locals (aligned)
    mov DWORD PTR [rbp-4], 10   ; x = 10
    ; ... function body ...
    leave                 ; Restore stack (mov rsp, rbp; pop rbp)
    ret                   ; Pop return address and jump
```

### Stack Size

```bash
# View stack size limit
ulimit -s              # In KB (typically 8192 KB = 8 MB)

# Set stack size
ulimit -s 16384        # 16 MB

# View thread stack size
cat /proc/$$/limits | grep stack

# In C, check stack size
#include <sys/resource.h>
struct rlimit rl;
getrlimit(RLIMIT_STACK, &rl);
printf("Stack limit: %ld\n", rl.rlim_cur);
```

### Stack Overflow

Occurs when stack grows beyond allocated size:

```c
// Causes stack overflow
void recursive() {
    char large[1000000];  // 1 MB local array
    recursive();          // Infinite recursion
}

// Also causes overflow
void deep_recursion(int n) {
    if (n == 0) return;
    deep_recursion(n - 1);  // Deep recursion without base case
}
```

**Detection:**
```bash
# Enable stack protection (compile time)
gcc -fstack-protector-all program.c

# Runtime detection
dmesg | grep segfault
```

### Thread Stacks

Each thread has its own stack:

```c
#include <pthread.h>

void* thread_func(void* arg) {
    int local = 42;  // Each thread has its own 'local'
    return NULL;
}

int main() {
    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_func, NULL);
    pthread_create(&t2, NULL, thread_func, NULL);

    // Set thread stack size
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 2 * 1024 * 1024); // 2 MB
    pthread_create(&t2, &attr, thread_func, NULL);
}
```

**View thread stacks:**
```bash
cat /proc/$$/maps | grep stack
# [stack]           <- Main thread stack
# [stack:1234]      <- Thread 1234's stack
# [stack:1235]      <- Thread 1235's stack
```

### Stack Canaries

Protection against buffer overflows:

```c
gcc -fstack-protector-all program.c

// The compiler inserts:
// - Canary value before return address
// - Check before function returns
// - Abort if canary is corrupted

void vulnerable() {
    char buffer[64];
    strcpy(buffer, user_input);  // If overflow, canary detects it
}
```

---

## Heap Management

### Heap Fundamentals

The heap is used for **dynamic memory allocation**:
- Grows upward (low address → high address)
- Managed explicitly by programmer
- Allocated via `malloc()`, `calloc()`, `realloc()`
- Freed via `free()`
- More flexible but slower than stack

### Memory Allocation

```c
#include <stdlib.h>

// Allocate memory
int* ptr = malloc(100 * sizeof(int));    // 400 bytes
if (ptr == NULL) {
    // Allocation failed
}

// Allocate and zero-initialize
int* ptr2 = calloc(100, sizeof(int));    // 400 bytes, zeroed

// Resize allocation
ptr = realloc(ptr, 200 * sizeof(int));   // 800 bytes

// Free memory
free(ptr);
ptr = NULL;  // Good practice
```

### System Calls: brk() and sbrk()

`malloc()` uses `brk()`/`sbrk()` for small allocations:

```c
#include <unistd.h>

// Get current heap end (program break)
void* current_brk = sbrk(0);

// Increase heap by 1024 bytes
void* new_mem = sbrk(1024);

// Set heap end directly
brk(new_address);
```

**Process:**
1. `malloc()` requests memory from heap
2. If heap is too small, `malloc()` calls `sbrk()` to extend heap
3. `sbrk()` system call moves the program break
4. Kernel allocates more pages to the process

### System Call: mmap()

For large allocations (typically >128 KB), `malloc()` uses `mmap()`:

```c
#include <sys/mman.h>

// Allocate 1 MB anonymously
void* ptr = mmap(NULL, 1024*1024,
                 PROT_READ|PROT_WRITE,
                 MAP_PRIVATE|MAP_ANONYMOUS,
                 -1, 0);

if (ptr == MAP_FAILED) {
    perror("mmap");
}

// Free memory
munmap(ptr, 1024*1024);
```

**Why mmap for large allocations?**
- Can return memory to OS immediately
- Doesn't fragment heap
- Better for sparse access patterns

### Heap Layout

The heap is divided into chunks:

```
┌────────────────────────────────┐
│  Chunk Header (metadata)       │  <- Size, flags, prev/next pointers
├────────────────────────────────┤
│  User Data                     │  <- Returned by malloc()
│  ...                           │
├────────────────────────────────┤
│  Chunk Header                  │
├────────────────────────────────┤
│  User Data                     │
│  ...                           │
└────────────────────────────────┘
```

### Memory Allocators

#### glibc malloc (ptmalloc2)

Default allocator in Linux:

```c
// Uses bins (freelists) for different sizes:
// - Fast bins: 16-80 bytes (LIFO)
// - Small bins: <512 bytes (FIFO)
// - Large bins: ≥512 bytes (best fit)
// - Unsorted bin: recently freed chunks

// View malloc stats
malloc_stats();

// Configure malloc
mallopt(M_MMAP_THRESHOLD, 128*1024);  // mmap threshold
```

#### Alternative Allocators

```c
// jemalloc (used by Firefox, Redis)
// Install: apt-get install libjemalloc-dev
// Use: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so ./program

// tcmalloc (Google's allocator)
// Install: apt-get install libgoogle-perftools-dev
// Use: LD_PRELOAD=/usr/lib/libtcmalloc.so ./program
```

### Heap Visualization

```bash
# View heap size
cat /proc/$$/status | grep VmData

# View heap region
cat /proc/$$/maps | grep heap

# Analyze heap usage
valgrind --tool=massif ./program
ms_print massif.out.*

# Heap profiling
valgrind --tool=memcheck --leak-check=full ./program

# Real-time heap monitoring
heaptrack ./program
heaptrack_gui heaptrack.program.*.gz
```

### Common Heap Issues

#### 1. Memory Leak

```c
void leak() {
    int* ptr = malloc(100);
    // Forgot to free(ptr)
}  // Memory never freed

// Detection
valgrind --leak-check=full ./program
```

#### 2. Double Free

```c
int* ptr = malloc(100);
free(ptr);
free(ptr);  // ERROR: Double free

// Prevention
free(ptr);
ptr = NULL;  // Freeing NULL is safe
```

#### 3. Use After Free

```c
int* ptr = malloc(100);
free(ptr);
*ptr = 42;  // ERROR: Use after free
```

#### 4. Heap Fragmentation

```
Before:
[Used][Free  100KB  ][Used][Free  50KB  ][Used]

After many alloc/free:
[Used][Free 10KB][Used][Free 5KB][Used][Free 3KB]
              ↑ Can't allocate 50KB contiguous block
```

**Mitigation:**
- Use memory pools
- Custom allocators
- Minimize allocation/deallocation churn

### Heap Security

#### Heap Overflow

```c
int* arr = malloc(10 * sizeof(int));
arr[15] = 42;  // Overflow! Corrupts heap metadata

// Protection:
// - ASLR (Address Space Layout Randomization)
// - Heap canaries
// - Safe libraries (AddressSanitizer)
```

#### Heap Spraying

Attack technique filling heap with predictable data.

**Defense:**
```bash
# Enable ASLR
echo 2 > /proc/sys/kernel/randomize_va_space

# Compile with sanitizers
gcc -fsanitize=address program.c
```

---

## Shared Objects (SO)

### Dynamic Linking

Shared objects (.so files) are dynamically linked libraries:

```bash
# List shared library dependencies
ldd /bin/ls
# Output:
#   linux-vdso.so.1 (0x00007fff...)
#   libselinux.so.1 => /lib/x86_64-linux-gnu/libselinux.so.1
#   libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
#   /lib64/ld-linux-x86-64.so.2 (0x00007f...)

# Show shared objects loaded by process
cat /proc/$$/maps | grep "\.so"
pmap $$ | grep "\.so"
```

### Static vs Dynamic Linking

```bash
# Static linking (larger binary, no dependencies)
gcc -static program.c -o program_static

# Dynamic linking (smaller binary, needs .so files)
gcc program.c -o program_dynamic

# Compare sizes
ls -lh program_static program_dynamic
# static: ~800 KB, dynamic: ~16 KB
```

### Creating Shared Libraries

```c
// mylib.c
#include "mylib.h"

int add(int a, int b) {
    return a + b;
}
```

```c
// mylib.h
#ifndef MYLIB_H
#define MYLIB_H

int add(int a, int b);

#endif
```

```bash
# Compile as shared object
gcc -fPIC -c mylib.c -o mylib.o
gcc -shared -o libmylib.so mylib.o

# Use the library
gcc program.c -L. -lmylib -o program

# Run (need to set LD_LIBRARY_PATH)
LD_LIBRARY_PATH=. ./program
```

### Position Independent Code (PIC)

PIC allows code to run at any memory address:

```bash
# Compile with PIC
gcc -fPIC -c code.c

# Check if binary is PIC
readelf -h binary | grep Type
# Type: DYN (Shared object file)  <- PIC
# Type: EXEC (Executable file)    <- Not PIC
```

**Why PIC?**
- ASLR: Security feature randomizes library load addresses
- Sharing: Multiple processes share same physical memory for library code
- Can't share non-PIC code (different virtual addresses)

### Dynamic Loader

`ld.so` / `ld-linux.so` loads shared libraries at runtime:

```bash
# Show loader
/lib64/ld-linux-x86-64.so.2 --version

# Trace library loading
LD_TRACE_LOADED_OBJECTS=1 ./program  # Same as ldd

# Debug dynamic linking
LD_DEBUG=all ./program 2>debug.log
LD_DEBUG=libs ./program   # Show library search
LD_DEBUG=bindings ./program  # Show symbol binding
```

### Library Search Path

```bash
# Search order:
# 1. RPATH (embedded in binary)
# 2. LD_LIBRARY_PATH environment variable
# 3. /etc/ld.so.cache (built from /etc/ld.so.conf)
# 4. /lib, /usr/lib

# View RPATH
readelf -d program | grep RPATH

# Set RPATH at compile time
gcc program.c -Wl,-rpath=/opt/mylib -o program

# Update library cache
ldconfig

# View cached libraries
ldconfig -p | grep libssl
```

### Dynamic Loading at Runtime

```c
#include <dlfcn.h>

// Load library at runtime
void* handle = dlopen("libmylib.so", RTLD_LAZY);
if (!handle) {
    fprintf(stderr, "Error: %s\n", dlerror());
    exit(1);
}

// Get function pointer
typedef int (*add_func)(int, int);
add_func add = (add_func) dlsym(handle, "add");
if (!add) {
    fprintf(stderr, "Error: %s\n", dlerror());
    exit(1);
}

// Use function
int result = add(3, 4);

// Unload library
dlclose(handle);
```

```bash
# Compile with -ldl
gcc program.c -ldl -o program
```

### Symbol Resolution

```bash
# List symbols in library
nm -D libmylib.so
# T add  <- Defined in text (function)
# U printf  <- Undefined (needs to be resolved)

# List all symbols (including internal)
nm libmylib.so

# Show only exported symbols
objdump -T libmylib.so

# Symbol versioning
objdump -T /lib/x86_64-linux-gnu/libc.so.6 | grep printf
# printf@@GLIBC_2.2.5
```

### Preloading Libraries

```bash
# Inject library before all others
LD_PRELOAD=/path/to/mylib.so ./program

# Common use cases:
# 1. Override functions (malloc, free)
# 2. Instrumentation
# 3. Testing/mocking

# Example: Override malloc
cat > mymalloc.c << 'EOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

void* malloc(size_t size) {
    static void* (*real_malloc)(size_t) = NULL;
    if (!real_malloc)
        real_malloc = dlsym(RTLD_NEXT, "malloc");

    printf("malloc(%zu)\n", size);
    return real_malloc(size);
}
EOF

gcc -shared -fPIC mymalloc.c -o mymalloc.so -ldl
LD_PRELOAD=./mymalloc.so ls  # Traces all malloc calls
```

### Lazy Binding

Functions are resolved on first call, not at load time:

```bash
# Immediate binding (resolve all symbols at load)
LD_BIND_NOW=1 ./program

# Check binding
readelf -d program | grep BIND_NOW
```

---

## Process Lifecycle

### Process Creation

```
Parent Process
      |
      | fork()
      |
      +---> Creates copy
      |
   Parent            Child
   (returns          (returns
    child PID)        0)
```

### Process States

A process transitions through several states:

```
                   fork()
    [New] ────────────────> [Ready]
                                │
                                │ Scheduler selects
                                ↓
    [Terminated] <──────── [Running] ────────> [Waiting/Blocked]
                   exit()       ↑                    │
                                │                    │ I/O complete,
                                │                    │ event occurs
                                └────────────────────┘
```

**States:**
- **R (Running/Runnable)**: Executing or waiting for CPU
- **S (Sleeping)**: Waiting for an event (interruptible)
- **D (Disk Sleep)**: Waiting for I/O (uninterruptible)
- **T (Stopped)**: Stopped by signal (SIGSTOP, SIGTSTP)
- **Z (Zombie)**: Terminated but not reaped by parent
- **I (Idle)**: Kernel thread

```bash
# View process state
ps aux | awk '{print $8, $11}'
# S     /usr/bin/bash
# R+    ps aux
```

---

## Process Operations

### fork() - Create Child Process

```c
#include <unistd.h>
#include <sys/types.h>

pid_t pid = fork();

if (pid < 0) {
    // Fork failed
    perror("fork");
    exit(1);
} else if (pid == 0) {
    // Child process
    printf("Child: PID=%d, PPID=%d\n", getpid(), getppid());
} else {
    // Parent process
    printf("Parent: PID=%d, Child PID=%d\n", getpid(), pid);
}
```

**What fork() copies:**
- ✅ Code (shared, copy-on-write)
- ✅ Stack (copied)
- ✅ Heap (copied)
- ✅ Data/BSS (copied)
- ✅ File descriptors (shared)
- ✅ Signal handlers (copied)
- ❌ PID (different)
- ❌ PPID (different)
- ❌ Locks (not inherited)

### vfork() - Fast Fork

```c
pid_t pid = vfork();

if (pid == 0) {
    // Child: Don't modify memory!
    // Parent is suspended, memory is shared
    execve("/bin/ls", args, env);
    _exit(1);  // If exec fails
}
// Parent resumes here
```

**vfork() vs fork():**
- vfork(): Child shares memory with parent (no copy-on-write)
- Parent is suspended until child calls `exec()` or `_exit()`
- Faster but dangerous (easy to corrupt parent's memory)

### clone() - Create Thread/Process

```c
#define _GNU_SOURCE
#include <sched.h>

// Low-level system call
// fork() and pthread_create() use clone() internally

int clone(int (*fn)(void *), void *stack, int flags, void *arg);

// Flags determine what's shared:
// CLONE_VM: Share memory
// CLONE_FS: Share filesystem info
// CLONE_FILES: Share file descriptors
// CLONE_SIGHAND: Share signal handlers
// CLONE_THREAD: Place in same thread group
```

### exec() Family - Replace Process Image

```c
#include <unistd.h>

// Replace current process with new program
execl("/bin/ls", "ls", "-l", NULL);
execv("/bin/ls", args);
execle("/bin/ls", "ls", "-l", NULL, envp);
execve("/bin/ls", args, envp);  // System call
execlp("ls", "ls", "-l", NULL);  // Search PATH
execvp("ls", args);              // Search PATH

// If exec succeeds, this line never executes
perror("exec failed");
```

**Common pattern: fork() + exec()**

```c
pid_t pid = fork();

if (pid == 0) {
    // Child: execute new program
    execl("/bin/date", "date", NULL);
    perror("exec failed");
    exit(1);
}

// Parent continues
wait(NULL);
```

### exit() - Terminate Process

```c
#include <stdlib.h>
#include <unistd.h>

// Normal termination (calls atexit handlers, flushes buffers)
exit(0);

// Immediate termination (no cleanup)
_exit(0);

// Register cleanup function
atexit(cleanup_func);
```

### wait() / waitpid() - Wait for Child

```c
#include <sys/wait.h>

// Wait for any child
int status;
pid_t child_pid = wait(&status);

// Wait for specific child
pid_t pid = waitpid(child_pid, &status, 0);

// Non-blocking wait
pid_t pid = waitpid(-1, &status, WNOHANG);

// Check exit status
if (WIFEXITED(status)) {
    printf("Exit code: %d\n", WEXITSTATUS(status));
}
if (WIFSIGNALED(status)) {
    printf("Killed by signal: %d\n", WTERMSIG(status));
}
```

**Zombie prevention:**
```c
// Method 1: wait() for children
while (wait(NULL) > 0);

// Method 2: Ignore SIGCHLD
signal(SIGCHLD, SIG_IGN);

// Method 3: Handle SIGCHLD
void sigchld_handler(int sig) {
    while (waitpid(-1, NULL, WNOHANG) > 0);
}
signal(SIGCHLD, sigchld_handler);
```

### Process Priority

```c
#include <sys/resource.h>

// Get/set nice value (-20 to 19, lower = higher priority)
int nice_val = getpriority(PRIO_PROCESS, 0);
setpriority(PRIO_PROCESS, 0, 10);  // Needs privilege for <0

// Nice command
nice -n 10 ./program     // Run with lower priority
renice -n 5 -p 12345     // Change priority of running process
```

### Signals

```c
#include <signal.h>

// Send signal
kill(pid, SIGTERM);   // To process
kill(-pgid, SIGTERM); // To process group
killpg(pgid, SIGTERM); // To process group

// Signal handler
void handler(int sig) {
    printf("Received signal %d\n", sig);
}

signal(SIGINT, handler);      // Simple
sigaction(SIGINT, &act, NULL); // Advanced

// Common signals:
// SIGINT (2): Interrupt (Ctrl+C)
// SIGKILL (9): Kill (uncatchable)
// SIGTERM (15): Terminate
// SIGSTOP (19): Stop (uncatchable)
// SIGCONT (18): Continue
// SIGCHLD (17): Child terminated
```

---

## Process States

### State Transitions

```bash
# View state in real-time
top
htop

# Process state codes
ps aux
# D    Uninterruptible sleep (usually I/O)
# R    Running or runnable (on run queue)
# S    Interruptible sleep (waiting for event)
# T    Stopped (job control or debugger)
# W    Paging (not valid since 2.6.xx)
# X    Dead (should never be seen)
# Z    Zombie (terminated but not reaped)
# <    High priority
# N    Low priority
# L    Has pages locked into memory
# s    Is session leader
# l    Is multi-threaded
# +    In foreground process group
```

### Uninterruptible Sleep (D)

```bash
# Find processes in D state
ps aux | awk '$8 ~ /D/ {print}'

# Common causes:
# - Waiting for disk I/O
# - NFS hangs
# - Hardware issues

# Cannot be killed with SIGKILL!
```

### Zombie Processes

```bash
# Find zombies
ps aux | awk '$8 ~ /Z/ {print}'

# Parent's responsibility to reap
# If parent doesn't call wait(), child becomes zombie
# If parent dies, init adopts and reaps zombie

# Force parent to reap (send SIGCHLD)
kill -CHLD $PPID
```

---

## Process Management Tools

### ps - Process Status

```bash
# Most common usages
ps aux                   # All processes, user-oriented
ps -ef                   # All processes, full format
ps -eLf                  # Include threads
ps -p 1234               # Specific process
ps -u username           # User's processes
ps --forest              # Tree view
ps -o pid,ppid,cmd       # Custom columns

# Sort by CPU/memory
ps aux --sort=-%cpu | head
ps aux --sort=-%mem | head

# Watch specific process
watch -n 1 'ps -p 1234 -o pid,pcpu,pmem,cmd'
```

### top / htop - Interactive Monitor

```bash
# top
top
# Keys:
#   M: Sort by memory
#   P: Sort by CPU
#   k: Kill process
#   r: Renice process
#   1: Show individual CPUs
#   H: Show threads

# htop (more user-friendly)
htop
# Mouse-clickable
# F5: Tree view
# F6: Sort
# F9: Kill
```

### pgrep / pkill - Search/Kill by Name

```bash
# Find processes
pgrep firefox            # Print PIDs
pgrep -l firefox         # Print PIDs and names
pgrep -u username        # User's processes
pgrep -f "pattern"       # Match full command line

# Kill processes
pkill firefox            # Kill by name
pkill -9 firefox         # Force kill
pkill -u username        # Kill user's processes
```

### pstree - Process Tree

```bash
# View process hierarchy
pstree
pstree -p              # Show PIDs
pstree -p 1234         # Tree from specific process
pstree -s 1234         # Show parents

# Example output:
# systemd─┬─sshd───sshd───bash───vim
#         ├─apache2───10*[apache2]
#         └─nginx───4*[nginx]
```

### pidof - Find PID by Name

```bash
pidof firefox
pidof -s firefox      # Single PID only
```

### lsof - List Open Files

```bash
# Files opened by process
lsof -p 1234

# Processes with file open
lsof /var/log/syslog

# Network connections
lsof -i                # All
lsof -i :80           # Port 80
lsof -i TCP:80        # TCP port 80
lsof -i @192.168.1.1  # Remote host

# By user
lsof -u username
```

### strace - Trace System Calls

```bash
# Trace system calls
strace ./program
strace -p 1234        # Attach to running process

# Trace specific calls
strace -e open,read,write ./program
strace -e trace=network ./program
strace -e trace=process ./program

# Count calls
strace -c ./program

# Timestamp
strace -t ./program
strace -tt ./program  # Microseconds
strace -T ./program   # Time spent in each call

# Save to file
strace -o trace.log ./program

# Trace child processes
strace -f ./program
```

### ltrace - Trace Library Calls

```bash
# Trace library calls
ltrace ./program
ltrace -p 1234

# Specific library
ltrace -l libssl.so ./program

# Count calls
ltrace -c ./program
```

### gdb - Debugger

```bash
# Debug program
gdb ./program
gdb -p 1234           # Attach to running process

# Common commands
(gdb) run             # Start program
(gdb) break main      # Set breakpoint
(gdb) continue        # Continue execution
(gdb) next            # Step over
(gdb) step            # Step into
(gdb) backtrace       # Stack trace
(gdb) info threads    # List threads
(gdb) thread 2        # Switch to thread
(gdb) print var       # Print variable
(gdb) info proc mappings  # Memory map
```

### /proc Filesystem

See [/proc Filesystem](#proc-filesystem) section.

---

## /proc Filesystem

Virtual filesystem providing process and system information.

### Process Information

```bash
# Process directory: /proc/[pid]/

# Command line arguments
cat /proc/$$/cmdline | tr '\0' ' '

# Environment variables
cat /proc/$$/environ | tr '\0' '\n'

# Current working directory
ls -l /proc/$$/cwd

# Executable
ls -l /proc/$$/exe

# File descriptors
ls -l /proc/$$/fd/
# 0 -> /dev/pts/0 (stdin)
# 1 -> /dev/pts/0 (stdout)
# 2 -> /dev/pts/0 (stderr)

# Memory maps
cat /proc/$$/maps

# Memory statistics
cat /proc/$$/status
cat /proc/$$/statm

# System calls
cat /proc/$$/syscall

# Limits
cat /proc/$$/limits

# Stack trace
cat /proc/$$/stack

# Open files
ls -l /proc/$$/fd/

# Mount points
cat /proc/$$/mountinfo

# Namespace
ls -l /proc/$$/ns/
```

### /proc/[pid]/status

```bash
cat /proc/$$/status

# Key fields:
# Name: Process name
# State: Current state
# Tgid: Thread group ID
# Pid: Process ID
# PPid: Parent PID
# Uid: Real, Effective, Saved, Filesystem UIDs
# Gid: Real, Effective, Saved, Filesystem GIDs
# VmSize: Virtual memory size
# VmRSS: Resident Set Size (physical memory)
# VmData: Size of data segment
# VmStk: Size of stack
# VmExe: Size of text (code)
# VmLib: Shared library size
# Threads: Number of threads
# voluntary_ctxt_switches: Voluntary context switches
# nonvoluntary_ctxt_switches: Involuntary context switches
```

### /proc/[pid]/maps

```bash
cat /proc/$$/maps

# Format:
# address           perms offset  dev   inode   pathname
# 00400000-00401000 r-xp 00000000 08:01 123     /bin/bash

# Permissions:
# r: Read
# w: Write
# x: Execute
# p: Private (copy-on-write)
# s: Shared

# Special regions:
# [heap]
# [stack]
# [vdso]  - Virtual Dynamic Shared Object
# [vvar]  - Virtual variables
```

### System-wide Information

```bash
# CPU info
cat /proc/cpuinfo

# Memory info
cat /proc/meminfo

# Load average
cat /proc/loadavg

# Uptime
cat /proc/uptime

# Kernel version
cat /proc/version

# Filesystems
cat /proc/filesystems

# Devices
cat /proc/devices

# Interrupts
cat /proc/interrupts

# I/O ports
cat /proc/ioports

# Network
cat /proc/net/tcp
cat /proc/net/udp
cat /proc/net/unix
cat /proc/net/dev

# Block devices
cat /proc/diskstats
```

---

## Inter-Process Communication

### Pipes

```bash
# Anonymous pipe (shell)
ls | grep txt

# In C
int pipefd[2];
pipe(pipefd);
// pipefd[0]: read end
// pipefd[1]: write end
```

### Named Pipes (FIFOs)

```bash
# Create FIFO
mkfifo /tmp/mypipe

# Writer
echo "Hello" > /tmp/mypipe &

# Reader
cat /tmp/mypipe
```

### Message Queues

```c
#include <sys/msg.h>

// Create queue
int msgid = msgget(IPC_PRIVATE, 0666 | IPC_CREAT);

// Send
struct msgbuf {
    long mtype;
    char mtext[100];
};
msgsnd(msgid, &msg, sizeof(msg.mtext), 0);

// Receive
msgrcv(msgid, &msg, sizeof(msg.mtext), 1, 0);

// Delete
msgctl(msgid, IPC_RMID, NULL);
```

```bash
# View message queues
ipcs -q

# Remove queue
ipcrm -q <msqid>
```

### Shared Memory

```c
#include <sys/shm.h>

// Create shared memory
int shmid = shmget(IPC_PRIVATE, 4096, 0666 | IPC_CREAT);

// Attach
char* ptr = shmat(shmid, NULL, 0);

// Use
strcpy(ptr, "Hello");

// Detach
shmdt(ptr);

// Delete
shmctl(shmid, IPC_RMID, NULL);
```

```bash
# View shared memory
ipcs -m

# Remove shared memory
ipcrm -m <shmid>
```

### Semaphores

```c
#include <sys/sem.h>

// Create semaphore
int semid = semget(IPC_PRIVATE, 1, 0666 | IPC_CREAT);

// Initialize
semctl(semid, 0, SETVAL, 1);

// P (wait/acquire)
struct sembuf sb = {0, -1, 0};
semop(semid, &sb, 1);

// V (signal/release)
sb.sem_op = 1;
semop(semid, &sb, 1);

// Delete
semctl(semid, 0, IPC_RMID);
```

### Sockets

```c
// See src/linux/networking.md for details

// Unix domain socket (local IPC)
int sockfd = socket(AF_UNIX, SOCK_STREAM, 0);

// Internet socket
int sockfd = socket(AF_INET, SOCK_STREAM, 0);
```

### Signals

See [Signals](#signals) section.

---

## Advanced Patterns

### Copy-on-Write (COW)

After `fork()`, parent and child share memory pages:

```
Before fork():
Parent: [Page A]

After fork():
Parent: [Page A] ←─┐
                   │ Both point to same physical page
Child:  [Page A] ←─┘

After write by child:
Parent: [Page A]    ← Original page
Child:  [Page A']   ← New copy
```

**Benefits:**
- Fast fork() - no immediate copying
- Memory efficient - copy only modified pages
- View COW in action:

```bash
# Before fork
cat /proc/$$/status | grep VmRSS

# After fork (child shares memory)
# VmRSS doesn't double

# After child modifies memory
# VmRSS increases
```

### Virtual Memory

Process sees contiguous virtual address space:

```
Virtual Address Space         Physical Memory
┌────────────────┐            ┌────────────────┐
│  0xFFFFFFFF    │            │                │
│                │   ┌────────│  Frame 1234    │
│  [Stack]       │───┘        ├────────────────┤
│                │            │                │
│  ...           │   ┌────────│  Frame 5678    │
│  [Heap]        │───┘        ├────────────────┤
│                │            │                │
│  [Data/BSS]    │───┐        │  Frame 9012    │
│                │   └────────├────────────────┤
│  [Text]        │───┐        │                │
│  0x00000000    │   └────────│  Frame 3456    │
└────────────────┘            └────────────────┘

Page Table translates Virtual → Physical
```

**Page size:**
```bash
getconf PAGE_SIZE    # Usually 4096 bytes (4 KB)
```

### Context Switching

When CPU switches between processes:

1. **Save context** of current process:
   - Registers (PC, SP, etc.)
   - Process state

2. **Load context** of next process:
   - Restore registers
   - Switch page tables
   - Update kernel structures

**Cost:** Several microseconds

```bash
# Context switches per second
vmstat 1
# cs column shows context switches

# Per-process context switches
cat /proc/$$/status | grep ctxt
```

### Process Scheduling

Linux uses Completely Fair Scheduler (CFS):

```bash
# View scheduler
cat /proc/$$/sched

# Scheduling policies:
# SCHED_NORMAL (0): Default time-sharing
# SCHED_FIFO (1): Real-time FIFO
# SCHED_RR (2): Real-time round-robin
# SCHED_BATCH (3): Batch processing
# SCHED_IDLE (5): Very low priority

# Set policy
chrt -f 10 ./program     # FIFO, priority 10
chrt -r 10 ./program     # Round-robin
```

### Namespaces

Isolate processes (used by containers):

```bash
# Namespace types:
# - mnt: Mount points
# - pid: Process IDs
# - net: Network stack
# - ipc: IPC resources
# - uts: Hostname
# - user: User/group IDs
# - cgroup: Control groups

# View namespaces
ls -l /proc/$$/ns/

# Create namespace
unshare --pid --fork --mount-proc bash
# Now in new PID namespace, ps shows only local processes

# Enter namespace
nsenter -t <pid> -a  # All namespaces
```

### Control Groups (cgroups)

Limit/prioritize resources:

```bash
# v1 location
ls /sys/fs/cgroup/

# v2 location
ls /sys/fs/cgroup/unified/

# Create cgroup
mkdir /sys/fs/cgroup/memory/mygroup

# Set memory limit (100 MB)
echo 104857600 > /sys/fs/cgroup/memory/mygroup/memory.limit_in_bytes

# Add process to cgroup
echo $$ > /sys/fs/cgroup/memory/mygroup/cgroup.procs

# View cgroup of process
cat /proc/$$/cgroup
```

### Daemon Processes

Background services:

```c
// Daemonization steps
#include <unistd.h>
#include <sys/stat.h>

void daemonize() {
    // 1. Fork and exit parent
    pid_t pid = fork();
    if (pid > 0) exit(0);

    // 2. Create new session
    setsid();

    // 3. Fork again (prevent controlling terminal)
    pid = fork();
    if (pid > 0) exit(0);

    // 4. Change directory
    chdir("/");

    // 5. Close file descriptors
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    // 6. Set umask
    umask(0);

    // 7. Open log file
    open("/var/log/mydaemon.log", O_WRONLY|O_CREAT, 0644);
}
```

```bash
# Modern way (systemd)
systemctl start mydaemon
systemctl enable mydaemon
```

---

## Practical Examples

### Example 1: Memory Layout Viewer

```c
// memory_layout.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int global_init = 42;           // Data segment
int global_uninit;              // BSS segment
const int const_data = 100;     // Read-only data

void print_addresses() {
    int stack_var;                 // Stack
    int* heap_ptr = malloc(10);    // Heap

    printf("=== Memory Layout ===\n");
    printf("Text (function):    %p\n", (void*)print_addresses);
    printf("Data (initialized): %p\n", (void*)&global_init);
    printf("BSS (uninitialized):%p\n", (void*)&global_uninit);
    printf("Heap:               %p\n", (void*)heap_ptr);
    printf("Stack:              %p\n", (void*)&stack_var);
    printf("===================\n");

    free(heap_ptr);
}

int main() {
    printf("PID: %d\n", getpid());
    print_addresses();
    printf("\nCheck: cat /proc/%d/maps\n", getpid());
    sleep(30);  // Keep process alive
    return 0;
}
```

```bash
gcc memory_layout.c -o memory_layout
./memory_layout &
cat /proc/$(pgrep memory_layout)/maps
```

### Example 2: Fork and Exec

```c
// fork_exec.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    printf("Parent PID: %d\n", getpid());

    pid_t pid = fork();

    if (pid < 0) {
        perror("fork");
        exit(1);
    } else if (pid == 0) {
        // Child process
        printf("Child PID: %d, PPID: %d\n", getpid(), getppid());

        // Execute new program
        char* args[] = {"ls", "-l", NULL};
        execvp("ls", args);

        // Only reached if exec fails
        perror("exec");
        exit(1);
    } else {
        // Parent process
        printf("Parent created child: %d\n", pid);

        // Wait for child
        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            printf("Child exited with status: %d\n",
                   WEXITSTATUS(status));
        }
    }

    return 0;
}
```

### Example 3: Pipe Communication

```c
// pipe_example.c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int main() {
    int pipefd[2];
    char buffer[100];

    if (pipe(pipefd) == -1) {
        perror("pipe");
        exit(1);
    }

    pid_t pid = fork();

    if (pid == 0) {
        // Child: writer
        close(pipefd[0]);  // Close read end
        char* msg = "Hello from child!";
        write(pipefd[1], msg, strlen(msg) + 1);
        close(pipefd[1]);
        exit(0);
    } else {
        // Parent: reader
        close(pipefd[1]);  // Close write end
        read(pipefd[0], buffer, sizeof(buffer));
        printf("Parent received: %s\n", buffer);
        close(pipefd[0]);
        wait(NULL);
    }

    return 0;
}
```

### Example 4: Shared Memory

```c
// shm_example.c
#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>

int main() {
    int shmid = shmget(IPC_PRIVATE, 4096, IPC_CREAT | 0666);

    pid_t pid = fork();

    if (pid == 0) {
        // Child: writer
        char* ptr = shmat(shmid, NULL, 0);
        strcpy(ptr, "Shared memory message!");
        shmdt(ptr);
        exit(0);
    } else {
        // Parent: reader
        wait(NULL);
        char* ptr = shmat(shmid, NULL, 0);
        printf("Read from shared memory: %s\n", ptr);
        shmdt(ptr);
        shmctl(shmid, IPC_RMID, NULL);
    }

    return 0;
}
```

### Example 5: Process Monitor

```bash
#!/bin/bash
# process_monitor.sh

PID=$1

if [ -z "$PID" ]; then
    echo "Usage: $0 <pid>"
    exit 1
fi

echo "Monitoring PID: $PID"
echo "================================"

while kill -0 $PID 2>/dev/null; do
    clear
    echo "=== Process Info ==="
    ps -p $PID -o pid,ppid,state,pcpu,pmem,vsz,rss,cmd

    echo -e "\n=== Memory Details ==="
    cat /proc/$PID/status | grep -E "VmSize|VmRSS|VmData|VmStk|VmExe"

    echo -e "\n=== Open Files ==="
    ls -l /proc/$PID/fd/ 2>/dev/null | wc -l

    echo -e "\n=== Threads ==="
    ls /proc/$PID/task/ 2>/dev/null | wc -l

    echo -e "\n=== Context Switches ==="
    cat /proc/$PID/status | grep ctxt

    sleep 2
done

echo "Process $PID terminated"
```

### Example 6: Signal Handling

```c
// signal_example.c
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <stdlib.h>

volatile sig_atomic_t keep_running = 1;

void signal_handler(int signum) {
    if (signum == SIGINT) {
        printf("\nReceived SIGINT (Ctrl+C)\n");
        keep_running = 0;
    } else if (signum == SIGTERM) {
        printf("\nReceived SIGTERM\n");
        keep_running = 0;
    }
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("PID: %d\n", getpid());
    printf("Press Ctrl+C to stop\n");

    while (keep_running) {
        printf("Running...\n");
        sleep(1);
    }

    printf("Exiting gracefully\n");
    return 0;
}
```

---

## Summary

### Key Takeaways

1. **Process Components:**
   - Code (text), Data, BSS, Heap (dynamic), Stack (local vars)
   - PID, PPID, PGID, SID for identification
   - Shared objects for dynamic linking

2. **Memory Management:**
   - Stack: Automatic, grows down, fast, limited size
   - Heap: Manual, grows up, flexible, slower
   - Virtual memory with page tables
   - Copy-on-write optimization

3. **Process Operations:**
   - `fork()`: Create child process
   - `exec()`: Replace process image
   - `wait()`: Wait for child termination
   - `exit()`: Terminate process

4. **IPC Mechanisms:**
   - Pipes, message queues, shared memory
   - Semaphores, sockets, signals

5. **Tools:**
   - `ps`, `top`, `htop`: Process monitoring
   - `strace`, `ltrace`: System/library call tracing
   - `/proc` filesystem: Process introspection
   - `lsof`: Open files and connections

6. **Advanced:**
   - Namespaces: Process isolation
   - cgroups: Resource limiting
   - Scheduling policies and priorities

---

## References

- `man 2 fork`
- `man 2 exec`
- `man 2 wait`
- `man 7 signal`
- `man 5 proc`
- [Linux Kernel Documentation](https://www.kernel.org/doc/Documentation/)
- [The Linux Programming Interface](http://man7.org/tlpi/) by Michael Kerrisk
