# cscope

cscope is a developer's tool for browsing source code in a terminal environment. It's particularly useful for navigating large C codebases, allowing you to search for symbols, function calls, and definitions interactively.

## Overview

cscope builds a symbol database from source files and provides a text-based interface for code navigation. While originally designed for C, it also supports C++ and Java.

**Key Features:**
- Find function definitions and calls
- Search for symbols, assignments, and regular expressions
- Navigate to files containing specific text
- Interactive text-based interface
- Integration with text editors (Vim, Emacs)
- Cross-reference capabilities

## Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install cscope

# macOS
brew install cscope

# CentOS/RHEL
sudo yum install cscope

# Arch Linux
sudo pacman -S cscope

# Verify installation
cscope -V
```

## Basic Usage

### Building Database

```bash
# Build database from current directory
cscope -b

# Build database recursively
cscope -bR

# Build database from specific files
cscope -b file1.c file2.c file3.c

# Build from file list
find . -name "*.c" -o -name "*.h" > cscope.files
cscope -b

# Build without launching interface
cscope -b -q    # -q for faster database

# Update existing database
cscope -u -b
```

### Interactive Mode

```bash
# Launch cscope
cscope

# Launch with specific database
cscope -d    # Use existing database (don't rebuild)

# Launch recursively
cscope -R

# Launch in line-oriented mode
cscope -l
```

### Interactive Commands

```
# In cscope interface:

Tab         - Toggle between input field and results
Ctrl+D      - Exit cscope
Ctrl+P      - Navigate to previous result
Ctrl+N      - Navigate to next result
Enter       - View selected result
Space       - Display next page of results
1-9         - Edit file at result number

# Search types:
0 - Find this C symbol
1 - Find this global definition
2 - Find functions called by this function
3 - Find functions calling this function
4 - Find this text string
5 - Change this text string (grep pattern)
6 - Find this egrep pattern
7 - Find this file
8 - Find files #including this file
9 - Find assignments to this symbol
```

## Command Line Searches

```bash
# Find symbol
cscope -L0 symbol_name

# Find global definition
cscope -L1 function_name

# Find functions called by function
cscope -L2 function_name

# Find functions calling function
cscope -L3 function_name

# Find text string
cscope -L4 "error message"

# Find egrep pattern
cscope -L6 "struct.*{$"

# Find file
cscope -L7 filename.c

# Find files including header
cscope -L8 header.h

# Output to file
cscope -L0 main > results.txt
```

## Vim Integration

### Basic Setup

```vim
" Add to ~/.vimrc
if has("cscope")
    set csprg=/usr/bin/cscope
    set csto=0
    set cst
    set nocsverb
    " Add cscope database if it exists
    if filereadable("cscope.out")
        cs add cscope.out
    endif
    set csverb
endif
```

### Advanced Vim Configuration

```vim
" ~/.vimrc
if has("cscope")
    set csprg=/usr/bin/cscope
    set csto=0
    set cst
    set csverb

    " Load database
    if filereadable("cscope.out")
        cs add cscope.out
    elseif $CSCOPE_DB != ""
        cs add $CSCOPE_DB
    endif

    " Key mappings
    nmap <C-\>s :cs find s <C-R>=expand("<cword>")<CR><CR>
    nmap <C-\>g :cs find g <C-R>=expand("<cword>")<CR><CR>
    nmap <C-\>c :cs find c <C-R>=expand("<cword>")<CR><CR>
    nmap <C-\>t :cs find t <C-R>=expand("<cword>")<CR><CR>
    nmap <C-\>e :cs find e <C-R>=expand("<cword>")<CR><CR>
    nmap <C-\>f :cs find f <C-R>=expand("<cfile>")<CR><CR>
    nmap <C-\>i :cs find i ^<C-R>=expand("<cfile>")<CR>$<CR>
    nmap <C-\>d :cs find d <C-R>=expand("<cword>")<CR><CR>

    " Horizontal split
    nmap <C-@>s :scs find s <C-R>=expand("<cword>")<CR><CR>
    nmap <C-@>g :scs find g <C-R>=expand("<cword>")<CR><CR>
    nmap <C-@>c :scs find c <C-R>=expand("<cword>")<CR><CR>
endif

" Auto-rebuild cscope database
function! UpdateCscope()
    silent !cscope -Rb
    cs reset
endfunction
command! Cscope call UpdateCscope()
```

### Vim Commands

```vim
" In Vim:
:cs find s symbol          " Find symbol
:cs find g definition      " Find global definition
:cs find c function        " Find calls to function
:cs find t text           " Find text
:cs find e pattern        " Find egrep pattern
:cs find f file           " Find file
:cs find i file           " Find files #including file
:cs find d symbol         " Find functions called by symbol

" Show cscope connections
:cs show

" Reset cscope connections
:cs reset

" Kill cscope connection
:cs kill 0
```

## Advanced Usage

### Custom File Lists

```bash
# C/C++ project
find . \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) > cscope.files
cscope -b -q

# Exclude directories
find . -path "./build" -prune -o -name "*.c" -print > cscope.files

# Include specific directories only
find src include -name "*.[ch]" > cscope.files
cscope -b -q
```

### Kernel-style Setup

```bash
# Linux kernel style
cat << 'EOF' > build_cscope.sh
#!/bin/bash
LNX=/path/to/linux/source

find  $LNX                                                                \
    -path "$LNX/arch/*" ! -path "$LNX/arch/x86*" -prune -o              \
    -path "$LNX/tmp*" -prune -o                                          \
    -path "$LNX/Documentation*" -prune -o                                \
    -path "$LNX/scripts*" -prune -o                                      \
    -type f \( -name '*.[chxsS]' -o -name 'Makefile' \)                  \
    -print > cscope.files

cscope -b -q -k
EOF

chmod +x build_cscope.sh
./build_cscope.sh
```

### Multiple Projects

```bash
# Project 1
cd /project1
cscope -b -q
export CSCOPE_DB=/project1/cscope.out

# Project 2 (separate database)
cd /project2
cscope -b -q -f cscope_proj2.out

# Use in Vim
:cs add /project1/cscope.out /project1
:cs add /project2/cscope_proj2.out /project2
```

## Scripting with cscope

### Automated Searches

```bash
#!/bin/bash
# find_function_calls.sh

FUNC=$1

if [ -z "$FUNC" ]; then
    echo "Usage: $0 <function_name>"
    exit 1
fi

echo "Functions calling $FUNC:"
cscope -dL3 $FUNC

echo ""
echo "Functions called by $FUNC:"
cscope -dL2 $FUNC
```

### Generate Call Graph

```bash
#!/bin/bash
# Generate simple call graph

FUNC=$1

function recurse_calls() {
    local func=$1
    local indent=$2

    echo "${indent}${func}"

    # Find functions called by this function
    cscope -dL2 "$func" | while read line; do
        called=$(echo $line | awk '{print $2}')
        if [ ! -z "$called" ]; then
            recurse_calls "$called" "${indent}  "
        fi
    done
}

recurse_calls "$FUNC" ""
```

### Find Unused Functions

```bash
#!/bin/bash
# find_unused.sh

# Get all function definitions
cscope -dL1 "" | awk '{print $2}' | sort -u > /tmp/all_funcs.txt

# For each function, check if it's called
while read func; do
    if [ "$func" != "main" ]; then
        calls=$(cscope -dL3 "$func" | wc -l)
        if [ $calls -eq 0 ]; then
            echo "Unused: $func"
        fi
    fi
done < /tmp/all_funcs.txt

rm /tmp/all_funcs.txt
```

## Makefile Integration

```makefile
# Add to Makefile

.PHONY: cscope
cscope:
	@find . -name "*.[ch]" > cscope.files
	@cscope -b -q

.PHONY: cscope-clean
cscope-clean:
	@rm -f cscope.* cscope.files

.PHONY: cscope-update
cscope-update: cscope-clean cscope
```

## Configuration File

```bash
# ~/.cscoperc or project .cscoperc
# (cscope automatically loads this)

# Custom options (limited support)
# Most configuration done via command line
```

## Emacs Integration

```elisp
;; Add to ~/.emacs or ~/.emacs.d/init.el

(require 'xcscope)
(cscope-setup)

;; Key bindings
(define-key global-map [(control f3)] 'cscope-set-initial-directory)
(define-key global-map [(control f4)] 'cscope-find-this-symbol)
(define-key global-map [(control f5)] 'cscope-find-global-definition)
(define-key global-map [(control f6)] 'cscope-find-functions-calling-this-function)
(define-key global-map [(control f7)] 'cscope-find-called-functions)
(define-key global-map [(control f8)] 'cscope-find-this-text-string)
(define-key global-map [(control f9)] 'cscope-find-this-file)
(define-key global-map [(control f10)] 'cscope-find-files-including-file)

;; Auto-update database
(setq cscope-do-not-update-database nil)
```

## Best Practices

### Large Projects

```bash
# Build inverted index for faster searches
cscope -b -q

# Use compression for large databases
cscope -b -c

# Incremental updates
cscope -u -b -q

# Index only relevant files
find . -name "*.[ch]" \
    ! -path "*/test/*" \
    ! -path "*/build/*" \
    > cscope.files
cscope -b -q
```

### Project Setup Script

```bash
#!/bin/bash
# setup_cscope.sh

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$PROJECT_ROOT"

echo "Building cscope database for: $PROJECT_ROOT"

# Find relevant source files
find . \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" \) \
    ! -path "*/build/*" \
    ! -path "*/\.git/*" \
    ! -path "*/node_modules/*" \
    > cscope.files

# Build database with inverted index
cscope -b -q -k

echo "Database built: cscope.out"
echo ""
echo "Usage:"
echo "  cscope -d          # Launch interactive mode"
echo "  vim <file>         # Use with Vim (if configured)"
echo "  cscope -L0 symbol  # Command-line search"
```

### Automatic Rebuilds

```bash
# Add to project root
# .git/hooks/post-commit

#!/bin/bash
echo "Rebuilding cscope database..."
cscope -b -q -k
echo "Done"
```

## Common Patterns

### Search All Files

```bash
# Find all occurrences of a string
cscope -L4 "TODO"

# Find all error messages
cscope -L4 "error:"

# Find struct definitions
cscope -L6 "^struct"

# Find all malloc calls
cscope -L0 malloc
```

### Code Review

```bash
# Find all functions modified in recent commit
git diff --name-only HEAD~1 | grep '\.[ch]$' | while read file; do
    echo "=== $file ==="
    # Get function names from file
    ctags -x --c-kinds=f "$file" | awk '{print $1}'
done
```

## Troubleshooting

```bash
# Database not found
cscope -b -R    # Rebuild recursively

# Incomplete results
rm cscope.out*
cscope -b -q    # Rebuild with index

# Vim integration not working
:cs show        # Check connections
:cs reset       # Reset connections
:cs add cscope.out

# Permission denied
chmod 644 cscope.out*

# Slow searches
cscope -b -q    # Build with inverted index

# Wrong directory
export CSCOPE_DB=/path/to/cscope.out
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `cscope -b` | Build database |
| `cscope -R` | Recursive search |
| `cscope -d` | Use existing database |
| `cscope -u` | Update database |
| `cscope -q` | Build inverted index |
| `cscope -L0` | Find symbol |
| `cscope -L1` | Find definition |
| `cscope -L3` | Find callers |
| `:cs find s` | Vim: Find symbol |
| `:cs find g` | Vim: Find definition |

cscope is an essential tool for navigating large C codebases, providing fast symbol lookups and cross-references that make code exploration and maintenance significantly easier.
