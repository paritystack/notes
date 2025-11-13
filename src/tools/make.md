# make

make is a build automation tool that automatically builds executable programs and libraries from source code by reading files called Makefiles which specify how to derive the target program.

## Overview

make uses Makefiles to determine which parts of a program need to be recompiled and issues commands to rebuild them. It's particularly useful for managing dependencies in large projects.

**Key Concepts:**
- **Target**: The file to be created or action to be performed
- **Prerequisites**: Files that must exist before target can be built
- **Recipe**: Commands to create the target from prerequisites
- **Rule**: Combination of target, prerequisites, and recipe
- **Phony Target**: Target that doesn't represent a file

## Basic Makefile

### Simple Example

```makefile
# Basic Makefile structure
target: prerequisites
	recipe

# Example: Compile a C program
program: main.c
	gcc -o program main.c

# Clean up build artifacts
clean:
	rm -f program
```

### Running make

```bash
# Build default target (first target in Makefile)
make

# Build specific target
make clean

# Build multiple targets
make program test

# Show commands without executing
make -n

# Run with specific Makefile
make -f MyMakefile
```

## Makefile Syntax

### Basic Structure

```makefile
# Comments start with #

# Variable definition
CC = gcc
CFLAGS = -Wall -O2

# Rule with target, prerequisites, and recipe
program: main.o utils.o
	$(CC) -o program main.o utils.o

# Multiple recipes (each on new line, indented with TAB)
main.o: main.c
	@echo "Compiling main.c"
	$(CC) $(CFLAGS) -c main.c

# Target with no prerequisites
clean:
	rm -f *.o program
```

**Important:** Recipes must be indented with a TAB character, not spaces.

### Variables

```makefile
# Simple variable assignment
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -O2

# Recursive expansion (evaluated when used)
SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

# Simple expansion (evaluated immediately)
NOW := $(shell date)

# Conditional assignment (only if not set)
CC ?= gcc

# Append to variable
CFLAGS += -g

# Using variables
program: main.c
	$(CC) $(CFLAGS) -o program main.c
```

### Automatic Variables

```makefile
# $@ - Target name
# $< - First prerequisite
# $^ - All prerequisites
# $? - Prerequisites newer than target
# $* - Stem of pattern rule match

# Example usage
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
	# $< is the .c file
	# $@ is the .o file

program: main.o utils.o
	$(CC) -o $@ $^
	# $@ is 'program'
	# $^ is 'main.o utils.o'
```

## Pattern Rules

### Suffix Rules

```makefile
# Pattern rule for .c -> .o
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Pattern rule for .cpp -> .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Multiple wildcards
bin/%: src/%.c
	$(CC) $(CFLAGS) $< -o $@
```

### Wildcards

```makefile
# Wildcard function
SRCS = $(wildcard src/*.c)
OBJS = $(wildcard obj/*.o)

# Pattern substitution
OBJS = $(SRCS:.c=.o)
OBJS = $(SRCS:%.c=%.o)
OBJS = $(patsubst %.c,%.o,$(SRCS))

# Example
SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:.c=.o)
DEPS = $(SOURCES:.c=.d)
```

## Phony Targets

```makefile
# Declare phony targets
.PHONY: all clean install test

# Common phony targets
all: program library

clean:
	rm -f *.o *.d program

install: program
	cp program /usr/local/bin/

test: program
	./run_tests.sh

# Prevent make from checking if 'clean' file exists
.PHONY: clean
clean:
	rm -f *.o program
```

## C/C++ Project Examples

### Simple C Project

```makefile
# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g

# Target executable
TARGET = myprogram

# Source files
SRCS = main.c utils.c parser.c
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Link object files
$(TARGET): $(OBJS)
	$(CC) -o $@ $^

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
```

### C Project with Headers

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -O2 -Iinclude

SRCDIR = src
OBJDIR = obj
BINDIR = bin

SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/program

all: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	$(CC) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create directories if they don't exist
$(BINDIR) $(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean
```

### C++ Project with Libraries

```makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
LDFLAGS = -lpthread -lm

SRCDIR = src
OBJDIR = obj
BINDIR = bin
INCDIR = include

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
DEPS = $(OBJS:.o=.d)
TARGET = $(BINDIR)/program

all: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -I$(INCDIR) -MMD -MP -c $< -o $@

$(BINDIR) $(OBJDIR):
	mkdir -p $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Include dependency files
-include $(DEPS)

.PHONY: all clean
```

### Multi-target Project

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -O2

# Multiple programs
PROGRAMS = server client

all: $(PROGRAMS)

server: server.o network.o utils.o
	$(CC) -o $@ $^

client: client.o network.o
	$(CC) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(PROGRAMS)

.PHONY: all clean
```

## Advanced Features

### Conditional Statements

```makefile
# Check variable value
ifdef DEBUG
    CFLAGS += -g -DDEBUG
else
    CFLAGS += -O2
endif

# Conditional based on value
ifeq ($(CC),gcc)
    CFLAGS += -Wall
endif

ifneq ($(OS),Windows_NT)
    LDFLAGS += -lpthread
endif

# OS detection
UNAME := $(shell uname -s)
ifeq ($(UNAME),Linux)
    LDFLAGS += -lrt
endif
ifeq ($(UNAME),Darwin)
    LDFLAGS += -framework CoreFoundation
endif
```

### Functions

```makefile
# Substitution
SRCS = main.c utils.c parser.c
OBJS = $(SRCS:.c=.o)
OBJS = $(patsubst %.c,%.o,$(SRCS))

# Directory operations
DIRS = $(dir src/main.c include/utils.h)  # "src/ include/"
FILES = $(notdir src/main.c include/utils.h)  # "main.c utils.h"

# String manipulation
FILES = $(wildcard *.c)
NAMES = $(basename $(FILES))  # Remove extension
UPPERS = $(shell echo $(FILES) | tr a-z A-Z)

# Filtering
SRCS = main.c test.c utils.c
PROD_SRCS = $(filter-out test.c,$(SRCS))  # "main.c utils.c"
TEST_SRCS = $(filter test%,$(SRCS))  # "test.c"

# Shell commands
DATE := $(shell date +%Y%m%d)
GIT_HASH := $(shell git rev-parse --short HEAD)
```

### Include Directives

```makefile
# Include another makefile
include config.mk

# Include with error if missing
include required.mk

# Include without error if missing
-include optional.mk

# Include all dependency files
-include $(DEPS)

# Example: config.mk
# CC = gcc
# CFLAGS = -Wall -O2
```

### Recursive Make

```makefile
# Top-level Makefile
SUBDIRS = lib src tests

all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

clean:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

.PHONY: all clean
```

### Dependency Generation

```makefile
CC = gcc
CFLAGS = -Wall -O2

SRCS = main.c utils.c
OBJS = $(SRCS:.c=.o)
DEPS = $(SRCS:.c=.d)

program: $(OBJS)
	$(CC) -o $@ $^

# Generate dependencies automatically
%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Include generated dependency files
-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS) program

.PHONY: clean
```

## Common Patterns

### Debug and Release Builds

```makefile
CC = gcc
CFLAGS = -Wall -Wextra

# Build modes
ifdef DEBUG
    CFLAGS += -g -O0 -DDEBUG
    TARGET = program_debug
else
    CFLAGS += -O2 -DNDEBUG
    TARGET = program
endif

SRCS = main.c utils.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) program program_debug

# Usage: make DEBUG=1
.PHONY: all clean
```

### Installation Targets

```makefile
PREFIX = /usr/local
BINDIR = $(PREFIX)/bin
DATADIR = $(PREFIX)/share/myapp

all: program

program: main.o
	$(CC) -o $@ $^

install: program
	install -d $(BINDIR)
	install -m 755 program $(BINDIR)
	install -d $(DATADIR)
	install -m 644 data/* $(DATADIR)

uninstall:
	rm -f $(BINDIR)/program
	rm -rf $(DATADIR)

.PHONY: all install uninstall
```

### Test Targets

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -O2

SRCS = main.c utils.c
TEST_SRCS = test_utils.c
OBJS = $(SRCS:.c=.o)
TEST_OBJS = $(TEST_SRCS:.c=.o)

program: $(OBJS)
	$(CC) -o $@ $^

test_runner: $(TEST_OBJS) utils.o
	$(CC) -o $@ $^

test: test_runner
	./test_runner

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TEST_OBJS) program test_runner

.PHONY: test clean
```

### Static Library

```makefile
CC = gcc
AR = ar
CFLAGS = -Wall -Wextra -O2

LIBNAME = mylib
SRCS = lib1.c lib2.c lib3.c
OBJS = $(SRCS:.c=.o)
TARGET = lib$(LIBNAME).a

all: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	install -d /usr/local/lib
	install -m 644 $(TARGET) /usr/local/lib
	install -d /usr/local/include/$(LIBNAME)
	install -m 644 *.h /usr/local/include/$(LIBNAME)

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all install clean
```

### Shared Library

```makefile
CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC
LDFLAGS = -shared

LIBNAME = mylib
VERSION = 1.0.0
MAJOR = 1

SRCS = lib1.c lib2.c lib3.c
OBJS = $(SRCS:.c=.o)
TARGET = lib$(LIBNAME).so.$(VERSION)
SONAME = lib$(LIBNAME).so.$(MAJOR)
LINKNAME = lib$(LIBNAME).so

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -Wl,-soname,$(SONAME) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

install: $(TARGET)
	install -d /usr/local/lib
	install -m 755 $(TARGET) /usr/local/lib
	ln -sf $(TARGET) /usr/local/lib/$(SONAME)
	ln -sf $(SONAME) /usr/local/lib/$(LINKNAME)
	ldconfig

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all install clean
```

## Make Options

### Common Flags

```bash
# Run in parallel (4 jobs)
make -j4

# Keep going on errors
make -k

# Show commands without executing
make -n
make --dry-run

# Print directory changes
make -w

# Ignore errors
make -i

# Touch files instead of building
make -t

# Print database of rules
make -p

# Treat warnings as errors
make --warn-undefined-variables
```

### Environment Variables

```bash
# Override variables
make CC=clang CFLAGS="-O3"

# Use specific Makefile
make -f Makefile.custom

# Change directory
make -C src/

# Set variables in Makefile
export CC=gcc
make
```

## Best Practices

### Structure and Organization

```makefile
# 1. Use variables for configurability
CC = gcc
CFLAGS = -Wall -Wextra -O2
PREFIX = /usr/local

# 2. Declare phony targets
.PHONY: all clean install test

# 3. Use automatic variables
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 4. Add help target
help:
	@echo "Available targets:"
	@echo "  all      - Build the program"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install the program"
	@echo "  test     - Run tests"

# 5. Use default goal
.DEFAULT_GOAL := all
```

### Dependency Management

```makefile
# Auto-generate dependencies
CC = gcc
CFLAGS = -Wall -O2
DEPFLAGS = -MMD -MP

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)
DEPS = $(SRCS:.c=.d)

%.o: %.c
	$(CC) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

-include $(DEPS)

clean:
	rm -f $(OBJS) $(DEPS)
```

### Error Handling

```makefile
# Stop on first error (default behavior)
.POSIX:

# Check for required tools
CHECK_CC := $(shell command -v $(CC) 2> /dev/null)
ifndef CHECK_CC
    $(error $(CC) not found in PATH)
endif

# Validate variables
ifndef TARGET
    $(error TARGET is not defined)
endif

# Conditional compilation
program: main.o
ifeq ($(CC),)
	$(error CC is not set)
endif
	$(CC) -o $@ $^
```

### Silent and Verbose Modes

```makefile
# Silent mode (suppress echo of commands)
.SILENT:

# Selective silence
all:
	@echo "Building..."
	$(CC) -o program main.c

# Verbose mode controlled by variable
ifdef VERBOSE
    Q =
else
    Q = @
endif

%.o: %.c
	@echo "CC $<"
	$(Q)$(CC) $(CFLAGS) -c $< -o $@
```

## Troubleshooting

### Common Issues

```bash
# "Missing separator" error
# Problem: Using spaces instead of TAB in recipe
# Solution: Ensure recipes are indented with TAB

# "No rule to make target" error
# Problem: Make can't find prerequisite file
make --debug=v  # Verbose debug output

# "Circular dependency" error
# Problem: Target depends on itself
# Solution: Review dependency chain

# Rebuild everything
make clean && make

# Show what make would do
make -n

# Print variables
make print-VARIABLE
```

### Debug Makefile

```makefile
# Print variable values
print-%:
	@echo $* = $($*)

# Usage: make print-CFLAGS

# Debug output
$(info Building with CC=$(CC))
$(warning This is a warning message)
$(error This stops the build)

# Show all variables
debug:
	@echo "SRCS = $(SRCS)"
	@echo "OBJS = $(OBJS)"
	@echo "CFLAGS = $(CFLAGS)"
```

### Performance Optimization

```bash
# Parallel builds
make -j$(nproc)  # Use all CPU cores

# Profile make execution
make -d > debug.log 2>&1

# Check which targets are rebuilt
make -d | grep "Must remake"

# Use ccache for faster compilation
CC = ccache gcc
```

## Complete Example

```makefile
# Project configuration
PROJECT = myapp
VERSION = 1.0.0

# Compiler settings
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -std=c11 -O2
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
LDFLAGS = -lm -lpthread

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin
TESTDIR = tests

# Files
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(SRCS:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
DEPS = $(OBJS:.o=.d)
TARGET = $(BINDIR)/$(PROJECT)

# Installation paths
PREFIX = /usr/local
BINPREFIX = $(PREFIX)/bin

# Build modes
ifdef DEBUG
    CFLAGS += -g -DDEBUG
    CXXFLAGS += -g -DDEBUG
endif

ifdef VERBOSE
    Q =
else
    Q = @
endif

# Targets
.PHONY: all clean install uninstall test help

all: $(TARGET)

$(TARGET): $(OBJS) | $(BINDIR)
	@echo "Linking $@"
	$(Q)$(CC) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	@echo "Compiling $<"
	$(Q)$(CC) $(CFLAGS) -I$(INCDIR) -MMD -MP -c $< -o $@

$(BINDIR) $(OBJDIR):
	$(Q)mkdir -p $@

clean:
	@echo "Cleaning build artifacts"
	$(Q)rm -rf $(OBJDIR) $(BINDIR)

install: $(TARGET)
	@echo "Installing to $(BINPREFIX)"
	$(Q)install -d $(BINPREFIX)
	$(Q)install -m 755 $(TARGET) $(BINPREFIX)

uninstall:
	@echo "Uninstalling from $(BINPREFIX)"
	$(Q)rm -f $(BINPREFIX)/$(PROJECT)

test: $(TARGET)
	@echo "Running tests"
	$(Q)./$(TESTDIR)/run_tests.sh

help:
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install the program"
	@echo "  uninstall - Uninstall the program"
	@echo "  test      - Run tests"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Build modes:"
	@echo "  make DEBUG=1    - Build with debug symbols"
	@echo "  make VERBOSE=1  - Show full commands"

-include $(DEPS)
```

## Useful Tips

1. **Always use `.PHONY`** for non-file targets
2. **Use automatic variables** (`$@`, `$<`, `$^`) for maintainability
3. **Generate dependencies** automatically with `-MMD -MP`
4. **Support parallel builds** with `make -j`
5. **Use variables** for all configuration options
6. **Include help target** for user guidance
7. **Handle errors gracefully** with proper checks
8. **Keep Makefiles readable** with comments and organization

make simplifies building complex projects by managing dependencies and minimizing rebuild time, making it an essential tool for C/C++ development and beyond.
