# Binary Analysis and Debugging Tools

A comprehensive guide to essential tools for analyzing, debugging, and understanding compiled binaries, shared libraries, and executables on Linux and Unix systems.

## Overview

Binary analysis tools help developers understand compiled programs, debug issues, analyze dependencies, and reverse engineer executables. These tools are essential for systems programming, security research, performance analysis, and troubleshooting deployment issues.

**Categories:**
- **Disassemblers**: objdump, gdb
- **Symbol Analysis**: nm, c++filt, addr2line
- **Binary Information**: file, readelf, size
- **String Extraction**: strings
- **Dependency Analysis**: ldd, patchelf
- **Runtime Tracing**: strace, ltrace
- **Hex Viewers**: hexdump, xxd, od
- **Binary Manipulation**: strip, objcopy, patchelf

## objdump - Object File Dumper

`objdump` displays information about object files, executables, and shared libraries. It's one of the most versatile binary analysis tools.

### Basic Usage

```bash
# Display file headers
objdump -f binary

# Display section headers
objdump -h binary

# Display all headers
objdump -x binary

# Disassemble executable sections
objdump -d binary

# Disassemble all sections (including data)
objdump -D binary

# Display source code intermixed with assembly
objdump -S binary

# Display full contents of all sections
objdump -s binary
```

### Disassembly Options

```bash
# Disassemble specific section
objdump -d -j .text binary

# Disassemble with source code (requires -g compilation)
objdump -S binary

# Intel syntax instead of AT&T
objdump -M intel -d binary

# Show line numbers
objdump -l -d binary

# Disassemble specific function
objdump -d binary | grep -A 50 '<function_name>'

# Start disassembly at specific address
objdump --start-address=0x400500 -d binary

# Stop at specific address
objdump --stop-address=0x400600 -d binary

# Disassemble address range
objdump --start-address=0x400500 --stop-address=0x400600 -d binary
```

### Symbol and Relocation Information

```bash
# Display symbol table
objdump -t binary

# Display dynamic symbol table
objdump -T binary

# Display relocation entries
objdump -r binary

# Display dynamic relocation entries
objdump -R binary

# Demangle C++ symbols
objdump -C -t binary
```

### Advanced Usage

```bash
# Display private headers (ELF program headers)
objdump -p binary

# Display dynamic section
objdump -p binary | grep NEEDED

# Show file format specific info
objdump -i

# Display debugging information
objdump -g binary

# Complete information dump
objdump -x -d -C binary > analysis.txt
```

### Common Patterns

```bash
# Find all calls to a specific function
objdump -d binary | grep "call.*function_name"

# Find all string references
objdump -s -j .rodata binary | less

# Check if binary is position independent
objdump -p binary | grep -i "type.*dyn"

# Find entry point
objdump -f binary | grep start

# Analyze GOT (Global Offset Table)
objdump -R binary

# Check for security features
objdump -p binary | grep -i "stack\|nx\|pie"

# Compare two binaries
diff <(objdump -d binary1) <(objdump -d binary2)

# Extract specific function disassembly
objdump -d binary | sed -n '/<main>:/,/^$/p'
```

### Architecture-Specific Disassembly

```bash
# ARM architecture
objdump -m arm -D binary

# MIPS architecture
objdump -m mips -D binary

# PowerPC
objdump -m powerpc -D binary

# Show available architectures
objdump -i
```

## ldd - List Dynamic Dependencies

`ldd` prints shared library dependencies of executables and shared libraries.

### Basic Usage

```bash
# List all shared library dependencies
ldd binary

# Verbose output with symbol versioning
ldd -v binary

# Show unused direct dependencies
ldd -u binary

# Display data relocations
ldd -d binary

# Display both data and function relocations
ldd -r binary
```

### Common Patterns

```bash
# Check for missing libraries
ldd binary 2>&1 | grep "not found"

# Find library path
ldd binary | grep libname

# Check if statically linked
ldd binary
# Output: "not a dynamic executable" for static binaries

# Compare dependencies between versions
diff <(ldd binary1) <(ldd binary2)

# Find all dependencies recursively
ldd -v binary

# Check library versions
ldd binary | grep libc
```

### Security Considerations

```bash
# WARNING: Never run ldd on untrusted binaries!
# ldd executes the binary to determine dependencies

# Safer alternative using objdump
objdump -p binary | grep NEEDED

# Or use readelf
readelf -d binary | grep NEEDED

# Or use ld.so directly (safer)
LD_TRACE_LOADED_OBJECTS=1 /lib64/ld-linux-x86-64.so.2 ./binary
```

### Troubleshooting Library Issues

```bash
# Set LD_LIBRARY_PATH for testing
LD_LIBRARY_PATH=/custom/path ldd binary

# Check library search paths
ldconfig -v | grep libname

# Show library loading verbosely
LD_DEBUG=libs ./binary

# Debug symbol resolution
LD_DEBUG=symbols ./binary

# Debug all library operations
LD_DEBUG=all ./binary 2>&1 | less

# Find which package provides a library (Debian/Ubuntu)
dpkg -S /path/to/library.so

# Find which package provides a library (RHEL/CentOS)
rpm -qf /path/to/library.so
```

## nm - List Symbols

`nm` lists symbols from object files, executables, and libraries.

### Basic Usage

```bash
# List all symbols
nm binary

# List only external symbols
nm -g binary

# List only undefined symbols
nm -u binary

# List symbols with demangled C++ names
nm -C binary

# Display symbol sizes
nm -S binary

# Sort by address
nm -n binary

# Sort by size
nm --size-sort binary

# Display dynamic symbols only
nm -D binary
```

### Symbol Types

```text
Symbol Type Meanings:
A - Absolute symbol
B/b - Uninitialized data (BSS)
C - Common symbol
D/d - Initialized data
G/g - Initialized data for small objects
I - Indirect reference
N - Debug symbol
R/r - Read-only data
S/s - Uninitialized data for small objects
T/t - Text (code) section
U - Undefined symbol
V/v - Weak object
W/w - Weak symbol
? - Unknown type

Uppercase = global/external
Lowercase = local
```

### Common Patterns

```bash
# Find definition of a symbol
nm -A *.o | grep symbol_name

# Check if symbol is defined
nm binary | grep -w symbol_name

# List all undefined symbols (missing dependencies)
nm -u binary

# Find which object file defines a symbol
nm -A *.o | grep " T symbol_name"

# Check for duplicate symbols
nm -A *.o | sort -k3 | uniq -f2 -d

# List all functions (text symbols)
nm binary | grep " T "

# List all global variables
nm binary | grep " D \| B "

# Find symbols by pattern
nm binary | grep -i "pattern"

# Compare symbols between binaries
diff <(nm binary1 | sort) <(nm binary2 | sort)

# Check symbol visibility
nm -g binary | wc -l    # Count exported symbols

# Find large symbols
nm --size-sort -S binary | tail -20

# List symbols with addresses and sizes
nm -S -n binary

# Check for C++ name mangling
nm binary | grep "_Z"
```

### Working with Archives

```bash
# List symbols from static library
nm libstatic.a

# List symbols with archive member names
nm -A libstatic.a

# Print index of archive
nm -s libstatic.a
```

## readelf - ELF File Reader

`readelf` displays detailed information about ELF (Executable and Linkable Format) files.

### Basic Usage

```bash
# Display ELF file header
readelf -h binary

# Display program headers
readelf -l binary

# Display section headers
readelf -S binary

# Display symbol table
readelf -s binary

# Display all headers
readelf -a binary

# Display dynamic section
readelf -d binary

# Display version information
readelf -V binary

# Display relocations
readelf -r binary
```

### Section Analysis

```bash
# Show section to segment mapping
readelf -l binary

# Display specific section
readelf -x .text binary

# Display section as strings
readelf -p .rodata binary

# Get section sizes
readelf -S binary | awk '{print $6, $7, $2}'

# Find sections by name
readelf -S binary | grep .data

# Display notes section
readelf -n binary
```

### Symbol Analysis

```bash
# Display symbol table with demangling
readelf -s -W binary | c++filt

# Display dynamic symbols only
readelf --dyn-syms binary

# Show symbol versions
readelf -V binary

# Display symbol by index
readelf -s binary | grep "\[13\]"

# Count symbols
readelf -s binary | wc -l
```

### Dynamic Analysis

```bash
# Show shared library dependencies
readelf -d binary | grep NEEDED

# Display RPATH and RUNPATH
readelf -d binary | grep PATH

# Show dynamic relocations
readelf -r binary

# Display PLT/GOT information
readelf -r binary | grep -E "PLT|GOT"

# Check for RELRO
readelf -l binary | grep GNU_RELRO

# Check for stack canary
readelf -s binary | grep __stack_chk_fail

# Check for PIE/PIC
readelf -h binary | grep Type
```

### Security Analysis

```bash
# Check for NX (No-Execute) stack
readelf -l binary | grep GNU_STACK

# Check for RELRO (Relocation Read-Only)
readelf -l binary | grep GNU_RELRO

# Check for PIE (Position Independent Executable)
readelf -h binary | grep "Type.*DYN"

# Check for FORTIFY_SOURCE
readelf -s binary | grep "__.*_chk"

# Display all security features
readelf -d -l binary | grep -E "BIND_NOW|RELRO|STACK"
```

### Common Patterns

```bash
# Find entry point
readelf -h binary | grep Entry

# Get architecture
readelf -h binary | grep Machine

# Check if stripped
readelf -S binary | grep -q .symtab && echo "Not stripped" || echo "Stripped"

# Find section addresses
readelf -S binary | awk '{print $5, $2}'

# Extract build ID
readelf -n binary | grep "Build ID"

# Display thread-local storage
readelf -l binary | grep TLS

# Show interpreter (dynamic linker)
readelf -l binary | grep interpreter

# Check ABI version
readelf -h binary | grep Version
```

## strings - Extract Printable Strings

`strings` finds printable character sequences in binary files.

### Basic Usage

```bash
# Extract all printable strings (default min length 4)
strings binary

# Set minimum string length
strings -n 8 binary

# Show file offset of each string
strings -t d binary      # Decimal offset
strings -t x binary      # Hexadecimal offset
strings -t o binary      # Octal offset

# Scan entire file (not just data sections)
strings -a binary

# Scan only data sections (default)
strings -d binary
```

### Common Patterns

```bash
# Find version strings
strings binary | grep -i version

# Find URLs
strings binary | grep -E "https?://"

# Find file paths
strings binary | grep "^/"

# Find email addresses
strings binary | grep "@"

# Find potential passwords or keys
strings binary | grep -i "password\|key\|secret"

# Find error messages
strings binary | grep -i "error\|warning\|failed"

# Search for specific string
strings binary | grep "search_term"

# Extract strings from core dump
strings core.dump | less

# Find function names
strings binary | grep "^[a-zA-Z_][a-zA-Z0-9_]*$"

# Look for debug strings
strings binary | grep -i "debug\|assert\|printf"

# Find SQL queries
strings binary | grep -i "SELECT\|INSERT\|UPDATE\|DELETE"

# Extract with context (combined with grep)
strings -n 6 binary | grep -i -C 2 "interesting"
```

### Encoding Options

```bash
# 7-bit ASCII (default)
strings -e s binary

# 8-bit ISO Latin-1
strings -e S binary

# 16-bit little-endian
strings -e l binary

# 16-bit big-endian
strings -e b binary

# All encodings
strings -e {s,S,l,b} binary
```

### Advanced Usage

```bash
# Combine with other tools
strings binary | sort | uniq
strings binary | grep -v "^[[:space:]]*$"  # Remove empty lines

# Compare strings between versions
diff <(strings binary1 | sort) <(strings binary2 | sort)

# Find hardcoded IP addresses
strings binary | grep -E "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b"

# Extract strings longer than 20 characters
strings -n 20 binary

# Save strings to file for analysis
strings -a -t x binary > strings_analysis.txt

# Find potential format string vulnerabilities
strings binary | grep "%s\|%x\|%n"
```

## file - File Type Identification

`file` determines file types based on magic numbers and content analysis.

### Basic Usage

```bash
# Identify file type
file binary

# Display MIME type
file -i binary
file --mime-type binary

# Brief mode (don't prepend filename)
file -b binary

# Dereference symlinks
file -L symlink

# Don't stop at first match
file -k binary

# Check magic file
file -m /path/to/magic binary
```

### Common Patterns

```bash
# Check if binary is stripped
file binary | grep stripped

# Find architecture
file binary | grep -o "x86-64\|i386\|ARM\|MIPS"

# Check if statically or dynamically linked
file binary | grep -o "statically\|dynamically"

# Identify all files in directory
file * | grep -v directory

# Find ELF files recursively
find . -type f -exec file {} \; | grep ELF

# Check if executable
file binary | grep executable

# Identify core dumps
file core.* | grep "core file"

# Check endianness
file binary | grep -o "LSB\|MSB"

# Batch check file types
find /path -type f -exec file -b {} \; | sort | uniq -c

# Find shared libraries
file /lib/* | grep "shared object"

# Identify debug binaries
file binary | grep "not stripped"
```

## strace - System Call Tracer

`strace` traces system calls and signals made by a process.

### Basic Usage

```bash
# Trace all system calls
strace ./program

# Attach to running process
strace -p PID

# Trace child processes
strace -f ./program

# Count system calls
strace -c ./program

# Save output to file
strace -o trace.log ./program

# Show timestamps
strace -t ./program       # Time of day
strace -tt ./program      # Microseconds
strace -T ./program       # Time spent in syscall
```

### Filtering System Calls

```bash
# Trace specific system call
strace -e open ./program
strace -e openat ./program

# Trace multiple system calls
strace -e open,read,write ./program

# Trace file operations
strace -e trace=file ./program

# Trace network operations
strace -e trace=network ./program

# Trace process operations
strace -e trace=process ./program

# Trace signals
strace -e trace=signal ./program

# Trace IPC operations
strace -e trace=ipc ./program

# Exclude system calls
strace -e \!write ./program
```

### Common Patterns

```bash
# Find which files a program opens
strace -e openat ./program 2>&1 | grep -E "\".*\""

# Debug library loading issues
strace -e open,openat,access ./program 2>&1 | grep "\.so"

# Find configuration files being read
strace -e openat,stat ./program 2>&1 | grep -E "\.conf|\.cfg|\.ini"

# Monitor network connections
strace -e socket,connect,sendto,recvfrom ./program

# Find why program hangs
strace -p PID

# Measure system call performance
strace -c -S calls ./program

# Debug file permission issues
strace -e open,openat,access ./program 2>&1 | grep EACCES

# Monitor child processes
strace -f -e clone,fork,vfork ./program

# Find files written
strace -e write,openat ./program 2>&1 | grep "O_WRONLY\|O_RDWR"

# Debug DNS resolution
strace -e socket,connect ./program 2>&1 | grep "AF_INET"

# Show string contents
strace -s 1024 ./program

# Trace only failed system calls
strace -Z ./program

# Monitor specific file
strace -e open,read,write -P /path/to/file ./program
```

### Advanced Usage

```bash
# Attach to all threads of a process
strace -p PID -f

# Trace with full string output
strace -s 4096 ./program

# Timestamp with relative times
strace -r ./program

# Filter by return value
strace -e open -e status=successful ./program
strace -e open -e status=failed ./program

# Quiet mode (suppress attach/detach messages)
strace -q -p PID

# Decode structures
strace -v ./program

# Output to separate files for each process
strace -ff -o trace ./program
# Creates trace.PID files
```

## ltrace - Library Call Tracer

`ltrace` traces library calls made by a program.

### Basic Usage

```bash
# Trace library calls
ltrace ./program

# Attach to running process
ltrace -p PID

# Follow forks
ltrace -f ./program

# Count library calls
ltrace -c ./program

# Save to file
ltrace -o trace.log ./program

# Show timestamps
ltrace -t ./program
ltrace -tt ./program      # Microseconds
```

### Filtering

```bash
# Trace specific function
ltrace -e malloc ./program

# Trace multiple functions
ltrace -e malloc,free ./program

# Exclude functions
ltrace -e \!printf ./program

# Trace functions from specific library
ltrace -l libssl.so ./program

# Show syscalls too
ltrace -S ./program
```

### Common Patterns

```bash
# Debug memory allocation
ltrace -e malloc,calloc,realloc,free ./program

# Monitor string operations
ltrace -e strcpy,strcat,strcmp ./program

# Track file operations
ltrace -e fopen,fread,fwrite,fclose ./program

# Debug crashes
ltrace -S -f ./program 2>&1 | tee crash.log

# Find memory leaks
ltrace -e malloc,free -c ./program

# Monitor network library calls
ltrace -e socket,connect,send,recv ./program

# Trace with full strings
ltrace -s 1024 ./program
```

## hexdump / xxd - Hex Viewers

Display files in hexadecimal and ASCII format.

### hexdump

```bash
# Canonical hex+ASCII display
hexdump -C file

# One-byte octal display
hexdump -b file

# Two-byte decimal display
hexdump -d file

# Two-byte hex display
hexdump -x file

# Custom format
hexdump -e '16/1 "%02x " "\n"' file

# Skip bytes
hexdump -s 1024 file

# Limit output
hexdump -n 512 file

# Display specific range
hexdump -s 1024 -n 512 file
```

### xxd

```bash
# Standard hex dump
xxd file

# Binary output
xxd -b file

# Plain hex dump (no addresses/ASCII)
xxd -p file

# Reverse hex dump (hex to binary)
xxd -r hexfile > binary

# Limit output
xxd -l 256 file

# Start at offset
xxd -s 1024 file

# Columns
xxd -c 32 file    # 32 bytes per line

# Include offset
xxd -o 0x1000 file
```

### Common Patterns

```bash
# Quick peek at binary file
xxd file | head

# Compare binary files
diff <(xxd file1) <(xxd file2)

# Extract magic number
xxd -l 16 file

# Search for hex pattern
xxd file | grep "pattern"

# Convert hex to binary
echo "48656c6c6f" | xxd -r -p

# Binary to hex
xxd -p file | tr -d '\n'

# Patch binary (modify specific bytes)
echo "00000000: 9090" | xxd -r - patched_file

# View memory dump
hexdump -C core.dump | less

# Compare checksums
xxd -p file | md5sum
```

## size - Section Sizes

Display section sizes and total size of object files.

### Basic Usage

```bash
# Display section sizes
size binary

# Berkeley format (default)
size -B binary

# SysV format (more detailed)
size -A binary

# Display in decimal (default)
size -d binary

# Display in octal
size -o binary

# Display in hexadecimal
size -x binary

# Total size only
size -t binary1 binary2 binary3
```

### Common Patterns

```bash
# Compare binary sizes
size binary1 binary2

# Sort by total size
size *.o | sort -k4 -n

# Find largest object files
size *.o | sort -k4 -rn | head

# Track size over commits
git log --oneline | head -10 | while read hash msg; do
    git checkout -q $hash
    echo -n "$hash: "
    size binary
done

# Check size limits
size binary | awk '{if ($4 > 1000000) print "Too large!"}'

# Compare sections
size -A binary | awk '{print $1, $2}'
```

## strip - Remove Symbols

Remove debugging symbols and symbol table from binaries.

### Basic Usage

```bash
# Strip all symbols
strip binary

# Strip debug symbols only
strip -g binary
strip --strip-debug binary

# Strip all debug and symbol info
strip --strip-all binary

# Keep specific symbols
strip --keep-symbol=main binary

# Strip into separate file
strip -o stripped_binary original_binary

# Preserve file timestamp
strip -p binary
```

### Common Patterns

```bash
# Check size reduction
ls -lh binary
strip binary
ls -lh binary

# Strip all binaries in directory
strip *.o

# Strip but keep debug info separate
objcopy --only-keep-debug binary binary.debug
strip binary
objcopy --add-gnu-debuglink=binary.debug binary

# Verify stripped status
file binary | grep stripped
readelf -S binary | grep symtab

# Strip specific sections
strip --remove-section=.comment binary
strip --remove-section=.note binary

# Batch strip with backup
for bin in *.o; do
    cp "$bin" "$bin.bak"
    strip "$bin"
done
```

## addr2line - Address to Line

Convert addresses to file names and line numbers.

### Basic Usage

```bash
# Convert address to source location
addr2line -e binary 0x400500

# Multiple addresses
addr2line -e binary 0x400500 0x400520

# Show function names
addr2line -f -e binary 0x400500

# Demangle C++ names
addr2line -C -f -e binary 0x400500

# Pretty print
addr2line -p -f -e binary 0x400500

# Use with backtrace
addr2line -e binary -f < backtrace.txt
```

### Common Patterns

```bash
# Decode stack trace from crash
grep "0x[0-9a-f]*" crash.log | addr2line -e binary -f -C

# Analyze core dump addresses
gdb -batch -ex "bt" -c core binary 2>&1 | \
    grep -oE "0x[0-9a-f]+" | \
    addr2line -e binary -f -p

# Continuous monitoring
tail -f error.log | while read line; do
    addr=$(echo "$line" | grep -oE "0x[0-9a-f]+")
    [ -n "$addr" ] && addr2line -e binary -f -C $addr
done

# Convert all addresses in file
grep -oE "0x[0-9a-f]+" addresses.txt | \
    xargs addr2line -e binary -f -C
```

## c++filt - Demangle C++ Symbols

Demangle C++ and Java symbol names.

### Basic Usage

```bash
# Demangle symbol
echo "_ZN9MyClass10myFunctionEv" | c++filt

# Demangle from nm output
nm binary | c++filt

# Demangle specific symbol
c++filt _ZN9MyClass10myFunctionEv

# Read from file
c++filt < mangled_symbols.txt
```

### Common Patterns

```bash
# Demangle nm output
nm -C binary  # Built-in demangling
nm binary | c++filt

# Demangle objdump output
objdump -t binary | c++filt

# Find mangled symbols
nm binary | grep "^_Z" | c++filt

# Compare mangled vs demangled
nm binary | grep "^_Z" | while read addr type sym; do
    echo "Mangled:   $sym"
    echo "Demangled: $(echo $sym | c++filt)"
    echo ""
done
```

## objcopy - Copy and Translate Objects

Copy and translate object files.

### Basic Usage

```bash
# Copy binary
objcopy input.o output.o

# Extract debug symbols
objcopy --only-keep-debug binary binary.debug

# Strip debug info
objcopy --strip-debug binary binary.stripped

# Add debug link
objcopy --add-gnu-debuglink=binary.debug binary

# Remove section
objcopy --remove-section=.comment binary

# Add section
objcopy --add-section .newsec=data.bin binary
```

### Common Patterns

```bash
# Split debug symbols
objcopy --only-keep-debug program program.debug
objcopy --strip-debug program
objcopy --add-gnu-debuglink=program.debug program

# Convert formats
objcopy -O binary input.elf output.bin
objcopy -I binary -O elf64-x86-64 data.bin data.o

# Extract section
objcopy -O binary --only-section=.text binary text.bin

# Change section attributes
objcopy --set-section-flags .data=alloc,load,readonly binary

# Embed file as binary data
objcopy -I binary -O elf64-x86-64 -B i386:x86-64 \
    --rename-section .data=.rodata,alloc,load,readonly,data,contents \
    data.bin data.o

# Create binary from ELF
objcopy -O binary program program.bin
```

## Integrated Workflows

### Analyzing Unknown Binary

```bash
# 1. Basic identification
file binary

# 2. Check if stripped
file binary | grep -q stripped

# 3. Check architecture and type
readelf -h binary

# 4. List dependencies
ldd binary
# or safer: readelf -d binary | grep NEEDED

# 5. Check security features
checksec binary  # If available
# or manually:
readelf -l binary | grep STACK
readelf -l binary | grep RELRO
readelf -h binary | grep Type

# 6. Extract strings
strings binary | less

# 7. List symbols (if not stripped)
nm -D binary | c++filt

# 8. Examine entry point and sections
readelf -l binary
readelf -S binary

# 9. Quick disassembly
objdump -d binary | less

# 10. Look for interesting functions
nm binary | grep -i "password\|auth\|key\|encrypt"
```

### Debugging Shared Library Issues

```bash
# 1. Check dependencies
ldd binary

# 2. Find missing libraries
ldd binary 2>&1 | grep "not found"

# 3. Check library paths
readelf -d binary | grep PATH

# 4. Trace library loading
LD_DEBUG=libs ./binary 2>&1 | tee lib_debug.log

# 5. Check symbol versions
readelf -V binary

# 6. Verify library exports expected symbols
nm -D /path/to/library.so | grep symbol_name

# 7. Check symbol binding
readelf -s binary | grep symbol_name

# 8. Trace runtime symbol resolution
LD_DEBUG=symbols ./binary 2>&1 | grep symbol_name
```

### Performance Analysis

```bash
# 1. Count system calls
strace -c ./program

# 2. Find slow operations
strace -T -e trace=all ./program 2>&1 | grep "<.*>"

# 3. Analyze I/O patterns
strace -e trace=file -c ./program

# 4. Monitor memory allocation
ltrace -e malloc,free -c ./program

# 5. Find hot functions
perf record ./program
perf report

# 6. Check binary size efficiency
size -A binary | sort -k2 -rn
```

### Reverse Engineering Workflow

```bash
# 1. Identify file
file binary

# 2. Extract strings for reconnaissance
strings -n 8 binary > strings.txt

# 3. Get symbol information
nm -C binary > symbols.txt 2>/dev/null || echo "Stripped"

# 4. Check imports/exports
readelf -s binary | grep FUNC > functions.txt

# 5. Disassemble main sections
objdump -M intel -d binary > disasm.txt

# 6. Examine ELF structure
readelf -a binary > elf_info.txt

# 7. Extract embedded data
objdump -s -j .rodata binary > rodata.txt

# 8. Analyze control flow
objdump -d binary | grep -E "call|jmp|ret"

# 9. Find cross-references
objdump -d binary | grep "call.*<function_name>"

# 10. Check for anti-debugging
strings binary | grep -i "ptrace\|debug\|trace"
readelf -s binary | grep ptrace
```

### Building Debug Package

```bash
# Extract debug symbols
objcopy --only-keep-debug program program.debug

# Strip original
objcopy --strip-debug --strip-unneeded program

# Add debug link
objcopy --add-gnu-debuglink=program.debug program

# Verify
file program.debug
file program
readelf -S program | grep debug_link

# Install debug symbols
sudo mkdir -p /usr/lib/debug
sudo cp program.debug /usr/lib/debug/

# Debug with separate symbols
gdb program
(gdb) info sources  # Should find debug symbols
```

## Tips and Best Practices

### General Tips

```bash
# Always verify file type first
file binary

# Use safer alternatives to ldd
readelf -d binary | grep NEEDED
objdump -p binary | grep NEEDED

# Combine tools for better analysis
nm -D binary | c++filt | grep "function_name"

# Save analysis to files
objdump -d binary > disasm.txt
readelf -a binary > elf_analysis.txt
strings binary > strings.txt

# Use grep for filtering
objdump -d binary | grep -A 10 "<main>:"

# Chain commands for complex queries
readelf -s binary | awk '{print $8}' | grep -v "^$" | sort | uniq
```

### Security Analysis

```bash
# Check for common security features
readelf -l binary | grep "GNU_STACK.*RWE"  # Should be RW, not RWE
readelf -l binary | grep GNU_RELRO
readelf -d binary | grep BIND_NOW
readelf -h binary | grep "Type.*DYN"  # PIE enabled

# Find dangerous functions
nm binary | grep -E "strcpy|strcat|sprintf|gets"
objdump -d binary | grep "call.*<strcpy@plt>"

# Check for hardcoded credentials
strings binary | grep -i "password\|passwd\|pwd"

# Look for format string vulnerabilities
strings binary | grep "%[0-9]"
```

### Debugging Workflow

```bash
# Quick crash analysis
file core.dump
gdb program core.dump
(gdb) bt
(gdb) info registers

# Find why binary won't run
ldd binary
strace ./binary 2>&1 | head -50

# Debug symbol resolution
LD_DEBUG=all ./binary 2>&1 | grep symbol_name

# Compare working vs broken binary
diff <(objdump -d working) <(objdump -d broken)
diff <(ldd working) <(ldd broken)
```

## Quick Reference

| Tool | Primary Use | Key Options |
|------|-------------|-------------|
| `objdump` | Disassembly, object file info | `-d`, `-S`, `-t`, `-T`, `-x` |
| `ldd` | Library dependencies | `-v`, `-u`, `-r` |
| `nm` | List symbols | `-C`, `-D`, `-u`, `-S`, `-A` |
| `readelf` | ELF file analysis | `-h`, `-l`, `-S`, `-s`, `-d` |
| `strings` | Extract strings | `-n`, `-t`, `-a` |
| `file` | File type identification | `-b`, `-i`, `-L` |
| `strace` | System call tracing | `-e`, `-p`, `-f`, `-c`, `-T` |
| `ltrace` | Library call tracing | `-e`, `-p`, `-f`, `-c`, `-S` |
| `hexdump` | Hex viewer | `-C`, `-n`, `-s` |
| `xxd` | Hex dump/reverse | `-p`, `-r`, `-l`, `-s` |
| `size` | Section sizes | `-A`, `-B`, `-t` |
| `strip` | Remove symbols | `-g`, `-s`, `-d` |
| `addr2line` | Address to line | `-e`, `-f`, `-C`, `-p` |
| `c++filt` | Demangle C++ | Input from pipe |
| `objcopy` | Manipulate objects | `--strip-debug`, `--add-section` |

## Common Use Cases

| Task | Command |
|------|---------|
| Find entry point | `readelf -h binary \| grep Entry` |
| List dependencies | `ldd binary` or `readelf -d binary \| grep NEEDED` |
| Check if stripped | `file binary \| grep stripped` |
| Disassemble function | `objdump -d binary \| grep -A 50 '<func>:'` |
| Find string references | `strings binary \| grep pattern` |
| Check architecture | `file binary` or `readelf -h binary` |
| List all symbols | `nm -D binary` |
| Trace file opens | `strace -e openat binary 2>&1` |
| Find library calls | `ltrace -e malloc,free binary` |
| Decode address | `addr2line -e binary -f -C 0x400500` |
| Check security features | `readelf -l binary \| grep STACK` |
| Extract section | `objcopy -O binary --only-section=.text binary out` |
| Compare binaries | `diff <(objdump -d bin1) <(objdump -d bin2)` |

These tools form the foundation of binary analysis and debugging on Unix-like systems. Mastering them enables effective troubleshooting, security analysis, and reverse engineering of compiled programs.
