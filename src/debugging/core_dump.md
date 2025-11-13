# Core Dump Analysis

Core dumps are memory snapshots of a process at the moment it crashed, essential for post-mortem debugging.

## Enable Core Dumps

```bash
# Check current limit
ulimit -c

# Enable unlimited core dumps
ulimit -c unlimited

# Make persistent (add to ~/.bashrc)
echo "ulimit -c unlimited" >> ~/.bashrc

# System-wide core dump configuration
sudo vim /etc/security/limits.conf
# Add: * soft core unlimited
```

## Configure Core Dump Location

```bash
# Set core dump pattern
sudo sysctl -w kernel.core_pattern=/tmp/core-%e-%p-%t

# Options:
# %e - executable name
# %p - PID
# %t - timestamp
# %s - signal number
# %h - hostname

# Or use systemd-coredump
sudo sysctl -w kernel.core_pattern=|/lib/systemd/systemd-coredump %P %u %g %s %t %c %h
```

## Generate Test Core Dump

```bash
# From running process
kill -SEGV <pid>

# From code
#include <signal.h>
raise(SIGSEGV);

# Trigger with gdb
gdb ./program
(gdb) run
(gdb) generate-core-file
```

## Analyze Core Dump with GDB

```bash
# Load core dump
gdb ./program core

# Or
gdb ./program core.12345

# GDB commands
(gdb) bt                 # Backtrace
(gdb) info threads       # List threads
(gdb) thread 2           # Switch to thread 2
(gdb) frame 0            # Select frame
(gdb) info locals        # Local variables
(gdb) print variable     # Print variable
(gdb) info registers     # CPU registers
(gdb) disassemble        # Disassemble current function
```

## Example Analysis Session

```gdb
$ gdb ./myapp core.12345
(gdb) bt
#0  0x00007f8b9c5a7428 in __GI_raise ()
#1  0x00007f8b9c5a902a in __GI_abort ()
#2  0x0000000000401234 in my_function () at myapp.c:42
#3  0x0000000000401567 in main () at myapp.c:100

(gdb) frame 2
(gdb) list
37      int *ptr = NULL;
38      int value = 0;
39
40      // This will crash
41      value = *ptr;
42
43      return value;

(gdb) print ptr
$1 = (int *) 0x0

(gdb) info locals
ptr = 0x0
value = 0
```

## Extract Information

```bash
# File information
file core.12345

# Strings in core
strings core.12345 | less

# Binary that produced core
file core.12345
# Look for "execfn:" in output

# All loaded libraries
gdb -batch -ex "info sharedlibrary" ./program core
```

## Automated Analysis

```bash
# Generate backtrace
gdb -batch -ex "bt" ./program core > backtrace.txt

# All threads backtrace
gdb -batch -ex "thread apply all bt" ./program core > all_threads.txt
```

## Core Dump with Containers

```bash
# Docker - enable core dumps
docker run --ulimit core=-1 myimage

# Kubernetes - configure pod
spec:
  containers:
  - name: myapp
    resources:
      limits:
        core: "-1"
```

## Best Practices

1. Always compile with debug symbols: `gcc -g`
2. Keep matching binaries for core analysis
3. Configure appropriate core dump location
4. Set reasonable ulimit to prevent disk filling
5. Use systemd-coredump for centralized management
6. Strip production binaries but keep debug symbols separate

Core dumps are invaluable for debugging crashes in production systems.
