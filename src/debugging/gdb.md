# GDB (GNU Debugger)

GDB is the GNU Project debugger, allowing you to see what is going on inside a program while it executes or what it was doing at the moment it crashed. It's an essential tool for debugging C, C++, and other compiled languages.

## Overview

GDB provides extensive facilities for tracing and altering program execution, including breakpoints, watchpoints, examining variables, and manipulating program state.

**Key Features:**
- Set breakpoints and watchpoints
- Step through code line by line
- Examine and modify variables
- Analyze core dumps
- Remote debugging
- Multi-threaded debugging
- Reverse debugging
- Python scripting support

## Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install gdb

# macOS (or use lldb)
brew install gdb

# CentOS/RHEL
sudo yum install gdb

# Arch Linux
sudo pacman -S gdb

# Verify installation
gdb --version
```

## Compiling for Debugging

```bash
# Compile with debug symbols (-g flag)
gcc -g program.c -o program
g++ -g program.cpp -o program

# Disable optimization for better debugging
gcc -g -O0 program.c -o program

# With all warnings
gcc -g -O0 -Wall -Wextra program.c -o program

# For C++ with debug symbols
g++ -g -std=c++17 program.cpp -o program
```

## Basic Usage

### Starting GDB

```bash
# Start GDB with program
gdb ./program

# With arguments
gdb --args ./program arg1 arg2

# Attach to running process
gdb -p <pid>
gdb attach <pid>

# Analyze core dump
gdb ./program core

# Quiet mode (no intro message)
gdb -q ./program
```

### Basic Commands

```gdb
# Running the program
(gdb) run                    # Start program
(gdb) run arg1 arg2          # Start with arguments
(gdb) start                  # Start and break at main()
(gdb) continue               # Continue execution (c)
(gdb) kill                   # Kill running program
(gdb) quit                   # Exit GDB (q)

# Breakpoints
(gdb) break main             # Break at function
(gdb) break main.c:42        # Break at line in file
(gdb) break *0x400500        # Break at address
(gdb) tbreak main            # Temporary breakpoint
(gdb) info breakpoints       # List breakpoints (info b)
(gdb) delete 1               # Delete breakpoint 1 (d 1)
(gdb) delete                 # Delete all breakpoints
(gdb) disable 1              # Disable breakpoint 1
(gdb) enable 1               # Enable breakpoint 1

# Stepping
(gdb) step                   # Step into (s)
(gdb) next                   # Step over (n)
(gdb) finish                 # Run until function returns
(gdb) until 50               # Run until line 50
(gdb) stepi                  # Step one instruction (si)
(gdb) nexti                  # Next instruction (ni)

# Examining code
(gdb) list                   # Show source code (l)
(gdb) list main              # List function
(gdb) list 42                # List around line 42
(gdb) disassemble            # Show assembly
(gdb) disassemble main       # Disassemble function

# Stack and frames
(gdb) backtrace              # Show call stack (bt)
(gdb) frame 0                # Switch to frame 0 (f 0)
(gdb) up                     # Move up stack frame
(gdb) down                   # Move down stack frame
(gdb) info frame             # Current frame info
(gdb) info args              # Function arguments
(gdb) info locals            # Local variables
```

## Examining Variables

```gdb
# Print variables
(gdb) print variable         # Print variable (p)
(gdb) print *pointer         # Dereference pointer
(gdb) print array[5]         # Array element
(gdb) print struct.member    # Structure member

# Different formats
(gdb) print/x variable       # Hexadecimal
(gdb) print/d variable       # Decimal
(gdb) print/t variable       # Binary
(gdb) print/c variable       # Character
(gdb) print/f variable       # Float
(gdb) print/s string_ptr     # String

# Display (auto-print on each stop)
(gdb) display variable       # Auto-display variable
(gdb) info display           # Show display list
(gdb) undisplay 1            # Remove display 1

# Examine memory
(gdb) x/10x $rsp             # Examine 10 hex words at stack pointer
(gdb) x/10i main             # Examine 10 instructions at main
(gdb) x/s string_ptr         # Examine string
(gdb) x/10b buffer           # Examine 10 bytes

# Format: x/[count][format][size] address
# Format: x=hex, d=decimal, i=instruction, s=string, c=char
# Size: b=byte, h=halfword, w=word, g=giant (8 bytes)

# Set variables
(gdb) set variable x = 42    # Set variable value
(gdb) set $i = 0             # Set convenience variable
```

## Watchpoints

```gdb
# Watch for changes
(gdb) watch variable         # Break when variable changes
(gdb) rwatch variable        # Break when variable is read
(gdb) awatch variable        # Break on read or write

# Conditional watchpoint
(gdb) watch x if x > 100

# Info and delete
(gdb) info watchpoints       # List watchpoints
(gdb) delete 2               # Delete watchpoint 2
```

## Conditional Breakpoints

```gdb
# Set conditional breakpoint
(gdb) break main.c:42 if x == 5

# Add condition to existing breakpoint
(gdb) condition 1 x == 5

# Remove condition
(gdb) condition 1

# Commands to execute at breakpoint
(gdb) commands 1
> print x
> continue
> end

# Ignore breakpoint N times
(gdb) ignore 1 10            # Ignore first 10 hits
```

## Thread Debugging

```gdb
# Thread information
(gdb) info threads           # List all threads
(gdb) thread 3               # Switch to thread 3
(gdb) thread apply all bt    # Backtrace all threads
(gdb) thread apply all print x

# Thread-specific breakpoints
(gdb) break main.c:42 thread 2

# Non-stop mode (continue while other threads stop)
(gdb) set non-stop on
```

## Core Dump Analysis

```bash
# Generate core dump
ulimit -c unlimited          # Enable core dumps

# Debug core dump
gdb ./program core

# In GDB
(gdb) bt                     # See where it crashed
(gdb) frame 0                # Examine crash frame
(gdb) print variable         # Check variable values
(gdb) info registers         # CPU registers at crash
```

## Advanced Features

### Reverse Debugging

```gdb
# Record execution
(gdb) record                 # Start recording
(gdb) record stop            # Stop recording

# Reverse execution
(gdb) reverse-step           # Step backward (rs)
(gdb) reverse-next           # Next backward (rn)
(gdb) reverse-continue       # Continue backward (rc)
(gdb) reverse-finish         # Reverse to function call
```

### Checkpoints

```gdb
# Save program state
(gdb) checkpoint             # Create checkpoint
(gdb) info checkpoints       # List checkpoints
(gdb) restart 1              # Restore checkpoint 1
(gdb) delete checkpoint 1    # Delete checkpoint
```

### Python Scripting

```gdb
# Python in GDB
(gdb) python print("Hello from GDB")

# Load Python script
(gdb) source script.py

# Python example
(gdb) python
> for i in range(5):
>     gdb.execute("print $i++")
> end
```

## GDB Configuration

### .gdbinit File

```bash
# ~/.gdbinit
set history save on
set history size 10000
set history filename ~/.gdb_history
set print pretty on
set print array on
set print array-indexes on
set python print-stack full

# Auto-load local .gdbinit
set auto-load safe-path /

# Custom commands
define phead
    print *($arg0)->head
end

define ptail
    print *($arg0)->tail
end
```

### GDB Dashboard

```bash
# Install GDB Dashboard
wget -P ~ https://git.io/.gdbinit

# Or with curl
curl -sSL https://git.io/.gdbinit > ~/.gdbinit

# Customization in ~/.gdbinit.d/init
```

## Common Patterns

### Debugging Segmentation Fault

```gdb
# Run program
(gdb) run

# When it crashes
Program received signal SIGSEGV, Segmentation fault.

# Check where it crashed
(gdb) backtrace

# Examine the failing instruction
(gdb) frame 0
(gdb) list

# Check variables
(gdb) print pointer
(gdb) print *pointer          # This might fail if NULL

# Check registers
(gdb) info registers
```

### Finding Memory Leaks

```gdb
# Set breakpoint at allocation
(gdb) break malloc
(gdb) commands
> backtrace
> continue
> end

# Set breakpoint at free
(gdb) break free
(gdb) commands
> backtrace
> continue
> end

# Or use Valgrind instead
```

### Debugging Infinite Loop

```gdb
# Start program
(gdb) run

# Interrupt (Ctrl+C)
^C
Program received signal SIGINT

# Check where it's stuck
(gdb) backtrace
(gdb) list

# Set breakpoint and check variable changes
(gdb) break main.c:loop_line
(gdb) commands
> print loop_var
> continue
> end
```

### Catching Signals

```gdb
# Catch specific signal
(gdb) catch signal SIGSEGV

# Catch all signals
(gdb) catch signal all

# Info signals
(gdb) info signals

# Handle signal (pass, nopass, stop, nostop, print, noprint)
(gdb) handle SIGINT nostop print pass
```

## Remote Debugging

### GDB Server

```bash
# On remote machine
gdbserver :1234 ./program

# Or attach to running process
gdbserver :1234 --attach <pid>

# On local machine
gdb ./program
(gdb) target remote remote-host:1234
(gdb) continue
```

### Serial/UART Debugging

```bash
# Connect via serial port
gdb ./program
(gdb) target remote /dev/ttyUSB0

# Set baud rate (if needed, in .gdbinit)
set serial baud 115200
```

## TUI Mode (Text User Interface)

```gdb
# Start TUI mode
(gdb) tui enable
(gdb) Ctrl+X A               # Toggle TUI

# TUI layouts
(gdb) layout src             # Source code
(gdb) layout asm             # Assembly
(gdb) layout split           # Source and assembly
(gdb) layout regs            # Registers

# Window focus
(gdb) focus cmd              # Focus command window
(gdb) focus src              # Focus source window

# Refresh display
(gdb) Ctrl+L                 # Refresh screen
```

## Useful Tricks

### Pretty Printing

```gdb
# Enable pretty printing
(gdb) set print pretty on
(gdb) set print array on
(gdb) set print array-indexes on

# STL pretty printers (C++)
(gdb) python
import sys
sys.path.insert(0, '/usr/share/gcc/python')
from libstdcxx.v6.printers import register_libstdcxx_printers
register_libstdcxx_printers(None)
end

# Now print STL containers nicely
(gdb) print my_vector
(gdb) print my_map
```

### Logging

```gdb
# Enable logging
(gdb) set logging on          # Logs to gdb.txt
(gdb) set logging file mylog.txt
(gdb) set logging overwrite on

# Log and display
(gdb) set logging redirect off
```

### Macros

```bash
# ~/.gdbinit
define plist
    set $node = $arg0
    while $node != 0
        print *$node
        set $node = $node->next
    end
end

# Usage
(gdb) plist head
```

### Function Breakpoints

```gdb
# Break on all functions matching pattern
(gdb) rbreak ^my_.*          # All functions starting with my_

# Break on exception throw (C++)
(gdb) catch throw

# Break on system calls
(gdb) catch syscall write
```

## Debugging Optimized Code

```gdb
# Problems with -O2, -O3
# Variables optimized away
# Inlining makes stepping difficult

# Solutions:
# 1. Compile with -Og (optimize for debugging)
gcc -g -Og program.c -o program

# 2. Disable specific optimizations
gcc -g -O2 -fno-inline program.c -o program

# 3. Use volatile for critical variables
volatile int debug_var;

# In GDB, skip inlined functions
(gdb) skip -rfu ^std::
```

## Integration with Other Tools

### Valgrind and GDB

```bash
# Run program under Valgrind with GDB server
valgrind --vgdb=yes --vgdb-error=0 ./program

# In another terminal
gdb ./program
(gdb) target remote | vgdb
```

### GDB with Make

```makefile
# Makefile
debug: program
	gdb ./program

.PHONY: debug
```

### GDB in VSCode

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "GDB Debug",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/program",
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "setupCommands": [
                {
                    "description": "Enable pretty-printing",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ]
        }
    ]
}
```

## Troubleshooting

```gdb
# Can't see source code
(gdb) directory /path/to/source

# Symbols not loaded
# Ensure compiled with -g
# Check symbols loaded
(gdb) info sources

# Can't set breakpoint
# Check function exists
(gdb) info functions pattern

# Program behavior different in GDB
# Try without breakpoints
# Timing-sensitive bugs

# GDB hangs
# Check for infinite loops in pretty printers
(gdb) set print elements 100

# Can't debug strip binary
# Need unstripped version or separate debug symbols
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `run` | Start program |
| `break` | Set breakpoint |
| `continue` | Continue execution |
| `step` | Step into |
| `next` | Step over |
| `print` | Print variable |
| `backtrace` | Show stack |
| `frame` | Select frame |
| `info locals` | Show local variables |
| `info args` | Show function arguments |
| `watch` | Set watchpoint |
| `list` | Show source code |
| `disassemble` | Show assembly |
| `quit` | Exit GDB |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Interrupt program |
| `Ctrl+D` | Exit GDB |
| `Enter` | Repeat last command |
| `Ctrl+X A` | Toggle TUI mode |
| `Ctrl+L` | Refresh screen |
| `Ctrl+P` | Previous command |
| `Ctrl+N` | Next command |

GDB is an indispensable tool for debugging compiled programs, offering powerful features for understanding program behavior, finding bugs, and analyzing crashes.
