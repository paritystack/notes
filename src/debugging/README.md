# Debugging

This directory contains guides for debugging software at various levels.

## Contents

- **[GDB](gdb.md)** - GNU Debugger for C/C++ applications
- **[LLDB](lldb.md)** - The LLVM debugger; GDB counterpart for Clang/macOS/Rust
- **[Binary Analysis Tools](tools.md)** - objdump, readelf, nm, strace, ltrace, and friends
- **[Valgrind](valgrind.md)** - Memory errors, leaks, races, and profiling via instrumentation
- **[Sanitizers](sanitizers.md)** - Compile-time ASan/UBSan/TSan/MSan runtime checks
- **[perf & Flame Graphs](perf_profiling.md)** - Sampling profiler and hardware counters
- **[rr (Record & Replay)](rr_debugging.md)** - Deterministic time-travel reverse debugging
- **[Core Dumps](core_dump.md)** - Analyzing program crashes
- **[Reverse Engineering](reverse_engineering.md)** - Ghidra, radare2, IDA; analysing binaries without source
- **[Debugging Methodology](debugging_methodology.md)** - Systematic approach, git bisect, delta debugging
- **[Linux Kernel](linux_kernel.md)** - Kernel-level debugging techniques

## Common Debugging Workflow

1. **Reproduce the issue** - Consistent reproduction is key
2. **Gather information** - Logs, error messages, core dumps
3. **Isolate the problem** - Narrow down the scope
4. **Form hypothesis** - What could cause this?
5. **Test hypothesis** - Use debuggers, logs, tests
6. **Fix and verify** - Implement fix and confirm

## Tools Overview

| Tool | Purpose | Level |
|------|---------|-------|
| gdb | Interactive debugging | Application |
| valgrind | Memory errors | Application |
| strace | System call tracing | Application/Kernel |
| ltrace | Library call tracing | Application |
| perf | Performance profiling | Application/Kernel |
| ftrace | Function tracing | Kernel |
| dmesg | Kernel messages | Kernel |

Effective debugging combines tools, techniques, and systematic thinking.
