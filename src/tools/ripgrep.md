# ripgrep (rg)

ripgrep is a line-oriented search tool that recursively searches your current directory for a regex pattern. It is extremely fast and respects your gitignore rules by default. ripgrep was created by Andrew Gallant (BurntSushi) as a faster, more user-friendly alternative to grep, ag (the silver searcher), and ack.

## Overview

ripgrep is built on top of Rust's regex engine and optimizes for speed without sacrificing usability. It's particularly well-suited for searching large codebases and respects common developer workflows.

**Key Features:**
- Extremely fast (often 2-10x faster than alternatives)
- Respects .gitignore and other ignore files by default
- Automatic recursive directory search
- Automatic skip of hidden files and binary files
- Smart case searching (case-insensitive if all lowercase, sensitive if mixed)
- Supports numerous text encodings (UTF-8, UTF-16, Latin-1, etc.)
- Parallel directory traversal and searching
- Powerful regex support with multiple regex engines
- Cross-platform (Linux, macOS, Windows, BSD)
- Compressed file search support
- Preprocessor support for custom file handling

**Common Use Cases:**
- Searching code repositories
- Finding text patterns in large codebases
- Grepping log files
- Code refactoring and analysis
- Security audits (finding API keys, passwords, etc.)
- Documentation searches
- Configuration file searches
- Quick file content exploration
- Build output analysis
- Data mining and text extraction

## Installation

```bash
# Debian/Ubuntu
sudo apt update
sudo apt install ripgrep

# RHEL/CentOS/Fedora
sudo dnf install ripgrep
# or
sudo yum install ripgrep

# Arch Linux
sudo pacman -S ripgrep

# macOS (Homebrew)
brew install ripgrep

# macOS (MacPorts)
sudo port install ripgrep

# Windows (Chocolatey)
choco install ripgrep

# Windows (Scoop)
scoop install ripgrep

# Windows (Winget)
winget install BurntSushi.ripgrep.MSVC

# Cargo (Rust package manager - any platform)
cargo install ripgrep

# From source (requires Rust)
git clone https://github.com/BurntSushi/ripgrep
cd ripgrep
cargo build --release
sudo cp target/release/rg /usr/local/bin/

# Verify installation
rg --version
```

## Basic Concepts

### How ripgrep Works

ripgrep operates in several phases:

1. **Pattern Compilation** - Compiles the regex pattern
2. **Directory Traversal** - Walks the directory tree (in parallel)
3. **File Filtering** - Applies ignore rules and file type filters
4. **File Searching** - Searches each file for matches (in parallel)
5. **Output Formatting** - Formats and displays results

### Smart Defaults

ripgrep comes with intelligent defaults that make it work well out of the box:

- **Recursive search** - Searches subdirectories automatically
- **Gitignore awareness** - Respects .gitignore, .ignore, .rgignore files
- **Hidden file skipping** - Skips hidden files and directories by default
- **Binary file skipping** - Automatically skips binary files
- **Smart case** - Case-insensitive if pattern is all lowercase, sensitive otherwise
- **Automatic encoding detection** - Handles UTF-8, UTF-16, etc.
- **Line buffering** - Optimized output for terminals and pipes

### Regex Syntax

By default, ripgrep uses Rust's regex engine which is similar to Perl-compatible regex (PCRE):

- `.` - Any character except newline
- `^` - Start of line
- `$` - End of line
- `*` - Zero or more repetitions
- `+` - One or more repetitions
- `?` - Zero or one repetition
- `{n,m}` - Between n and m repetitions
- `[abc]` - Character class (a, b, or c)
- `[^abc]` - Negated character class (not a, b, or c)
- `\d` - Digit
- `\w` - Word character
- `\s` - Whitespace
- `(...)` - Capturing group
- `|` - Alternation (or)
- `\b` - Word boundary

## Basic Operations

### Simple Search

```bash
# Search for pattern in current directory
rg "pattern"
rg "function"
rg "TODO"

# Search for exact string (no regex)
rg -F "exact.string.with.dots"
rg --fixed-strings "literal$string"

# Search in specific file
rg "pattern" file.txt

# Search in multiple files
rg "pattern" file1.txt file2.txt

# Search with multiple patterns (OR)
rg "pattern1|pattern2"
rg "error|warning|critical"

# Case-sensitive search
rg -s "Pattern"
rg --case-sensitive "CamelCase"

# Case-insensitive search (force)
rg -i "pattern"
rg --ignore-case "PATTERN"
```

### Recursive Search

```bash
# Search recursively in current directory (default)
rg "pattern"

# Search in specific directory
rg "pattern" /path/to/directory

# Search in multiple directories
rg "pattern" dir1/ dir2/ dir3/

# Limit recursion depth
rg --max-depth 2 "pattern"
rg --max-depth 1 "pattern"  # Only current directory

# Search without recursion
rg --max-depth 1 "pattern"
```

### File Type Filtering

```bash
# Search only in specific file types
rg -t py "pattern"          # Python files
rg -t js "pattern"          # JavaScript files
rg -t rust "pattern"        # Rust files
rg -t cpp "pattern"         # C++ files
rg -t java "pattern"        # Java files
rg -t go "pattern"          # Go files
rg -t md "pattern"          # Markdown files
rg -t html "pattern"        # HTML files
rg -t css "pattern"         # CSS files
rg -t json "pattern"        # JSON files

# Multiple file types
rg -t py -t js "pattern"
rg --type python --type javascript "pattern"

# Exclude file types
rg -T js "pattern"          # Exclude JavaScript files
rg --type-not javascript "pattern"

# List available file types
rg --type-list

# Add custom file type
rg --type-add 'custom:*.foo' -t custom "pattern"
```

### Glob Patterns

```bash
# Search files matching glob pattern
rg -g "*.py" "pattern"
rg --glob "*.js" "pattern"

# Multiple glob patterns
rg -g "*.{js,ts}" "pattern"
rg -g "*.py" -g "*.pyx" "pattern"

# Exclude with glob patterns
rg -g "!*.min.js" "pattern"
rg -g "!test*" "pattern"
rg --glob "!vendor/*" "pattern"

# Complex glob patterns
rg -g "src/**/*.rs" "pattern"
rg -g "**/test_*.py" "pattern"
```

### Basic Output Control

```bash
# Show line numbers (default)
rg "pattern"

# Hide line numbers
rg -N "pattern"
rg --no-line-number "pattern"

# Show column numbers
rg --column "pattern"

# Show only filenames with matches
rg -l "pattern"
rg --files-with-matches "pattern"

# Show only filenames without matches
rg --files-without-match "pattern"

# Count matches per file
rg -c "pattern"
rg --count "pattern"

# Count total matches
rg --count-matches "pattern"

# Show only matching part (not full line)
rg -o "pattern"
rg --only-matching "pattern"
```

## Advanced Searching

### Context Lines

```bash
# Show N lines after match
rg -A 3 "pattern"
rg --after-context 3 "pattern"

# Show N lines before match
rg -B 3 "pattern"
rg --before-context 3 "pattern"

# Show N lines before and after match
rg -C 3 "pattern"
rg --context 3 "pattern"

# Different before/after context
rg -B 5 -A 2 "pattern"
```

### Multiline Search

```bash
# Enable multiline mode
rg -U "pattern.*across.*lines"
rg --multiline "start.*\n.*middle.*\n.*end"

# Search for function definitions across lines
rg -U "function.*\{.*\n.*return"

# Find multi-line comments
rg -U "/\*.*\*/"

# Complex multiline patterns
rg -U "class \w+.*\n.*def __init__"
```

### Word Boundaries

```bash
# Match whole words only
rg -w "word"
rg --word-regexp "function"

# Match word boundaries with regex
rg "\bword\b"
rg "\bfunction\b"

# Combine with other options
rg -w -i "class"
```

### Replacement and Transformation

```bash
# Show replacements (doesn't modify files)
rg "old" -r "new"
rg "pattern" --replace "replacement"

# With capture groups
rg "(\w+)@(\w+)" -r '$2@$1'
rg "function (\w+)" -r 'def $1'

# Passthrough mode (prints all lines, highlighting matches)
rg --passthru "pattern"

# Passthrough with replacement
rg --passthru "old" -r "new"
```

### Hidden and Ignored Files

```bash
# Include hidden files
rg --hidden "pattern"
rg -. "pattern"

# Search all files (ignore .gitignore, .ignore, etc.)
rg -u "pattern"                    # Ignore .gitignore
rg -uu "pattern"                   # Ignore .gitignore and hidden files
rg -uuu "pattern"                  # Search everything (including binary)

# Don't respect ignore files
rg --no-ignore "pattern"

# Don't respect .gitignore
rg --no-ignore-vcs "pattern"

# Don't respect parent .gitignore
rg --no-ignore-parent "pattern"

# Don't skip hidden files
rg --hidden "pattern"
```

### Binary Files

```bash
# Search binary files
rg -a "pattern"
rg --text "pattern"

# Show binary file matches as hex
rg --binary "pattern"

# Skip binary files explicitly (default)
rg "pattern"

# Search binary with specific encoding
rg -E latin1 "pattern"
```

### Follow Symlinks

```bash
# Follow symbolic links
rg -L "pattern"
rg --follow "pattern"

# Default: don't follow symlinks
rg "pattern"
```

### Compressed Files

```bash
# Search in compressed files
rg -z "pattern"
rg --search-zip "pattern"

# Supported formats: gzip, bzip2, xz, lz4, lzma, zstd

# Search in .gz files
rg -z "pattern" logs.gz

# Search in multiple compressed files
rg -z "error" *.gz
```

## Pattern Matching

### Literal Strings

```bash
# Fixed string (no regex)
rg -F "string.with.dots"
rg -F "regex chars like * + ?"

# Useful for searching special characters
rg -F '$variable'
rg -F '[bracket]'
rg -F '(parenthesis)'
```

### Regular Expressions

```bash
# Basic regex
rg "fo+"                           # One or more 'o'
rg "colou?r"                       # Optional 'u'
rg "file\d+"                       # file followed by digits
rg "test_\w+"                      # test_ followed by word chars

# Character classes
rg "[aeiou]"                       # Any vowel
rg "[0-9]+"                        # One or more digits
rg "[A-Z][a-z]+"                   # Capital letter + lowercase

# Anchors
rg "^import"                       # Lines starting with import
rg "return$"                       # Lines ending with return
rg "^$"                            # Empty lines

# Word boundaries
rg "\bword\b"                      # Whole word
rg "\Btest"                        # Not at word boundary

# Alternation
rg "error|warning|fatal"
rg "(jpg|png|gif)$"

# Grouping
rg "func(tion)?"
rg "(get|set)_(\w+)"

# Quantifiers
rg "a{3}"                          # Exactly 3 'a's
rg "a{2,4}"                        # Between 2 and 4 'a's
rg "a{2,}"                         # 2 or more 'a's

# Lookahead (not in default engine, use -P)
rg -P "foo(?=bar)"                 # foo followed by bar
rg -P "foo(?!bar)"                 # foo not followed by bar

# Lookbehind (use -P)
rg -P "(?<=foo)bar"                # bar preceded by foo
rg -P "(?<!foo)bar"                # bar not preceded by foo
```

### PCRE2 Engine

```bash
# Use PCRE2 engine for advanced features
rg -P "pattern"
rg --pcre2 "pattern"

# Lookahead assertions
rg -P "password(?=.*[A-Z])"

# Lookbehind assertions
rg -P "(?<=\$)\d+"                 # Digits after $

# Recursive patterns
rg -P "\((?:[^()]++|(?R))*+\)"     # Balanced parentheses

# Named groups
rg -P "(?P<year>\d{4})-(?P<month>\d{2})"

# Conditionals
rg -P "(?(condition)yes-pattern|no-pattern)"
```

### Common Patterns

```bash
# Email addresses
rg "\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

# IP addresses
rg "\b(?:\d{1,3}\.){3}\d{1,3}\b"
rg "\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"

# URLs
rg "https?://[^\s]+"

# Phone numbers (US)
rg "\b\d{3}-\d{3}-\d{4}\b"
rg "\(\d{3}\)\s*\d{3}-\d{4}"

# UUIDs
rg "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

# Hex colors
rg "#[0-9a-fA-F]{6}\b"

# Credit card numbers (simple pattern)
rg "\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"

# Social Security Numbers
rg "\b\d{3}-\d{2}-\d{4}\b"

# Dates (YYYY-MM-DD)
rg "\b\d{4}-\d{2}-\d{2}\b"

# Dates (MM/DD/YYYY)
rg "\b\d{1,2}/\d{1,2}/\d{4}\b"

# Times (HH:MM:SS)
rg "\b\d{2}:\d{2}:\d{2}\b"

# MAC addresses
rg "([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})"

# IPv6 addresses
rg "([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"

# Base64 strings
rg "[A-Za-z0-9+/]{40,}={0,2}"
```

## Code Searching

### Function and Class Definitions

```bash
# Find function definitions (Python)
rg "^def \w+\("
rg "^\s*def \w+\("

# Find class definitions (Python)
rg "^class \w+"

# Find function definitions (JavaScript)
rg "function \w+\("
rg "const \w+ = \("

# Find function definitions (C/C++)
rg "^\w+\s+\w+\([^)]*\)\s*\{"

# Find class definitions (Java)
rg "^(public|private|protected)?\s*(class|interface) \w+"

# Find method definitions (Ruby)
rg "^\s*def \w+"

# Find function definitions (Go)
rg "^func \w+\("

# Find class definitions (Rust)
rg "^(pub\s+)?struct \w+"
```

### Import and Include Statements

```bash
# Python imports
rg "^import \w+"
rg "^from .* import"

# JavaScript/TypeScript imports
rg "^import .* from"
rg "^import\s+.*\s+from\s+"

# C/C++ includes
rg "^#include [<\"].*[>\"]"

# Java imports
rg "^import .*;"

# Go imports
rg "^import \("
```

### Variable and Constant Declarations

```bash
# JavaScript const/let/var
rg "^(const|let|var) \w+"

# Python variables (assignments at module level)
rg "^\w+ = "

# C/C++ variable declarations
rg "^(int|char|float|double|void|bool|auto) \w+"

# Java variable declarations
rg "^(private|public|protected)?\s*(static)?\s*(final)?\s*\w+ \w+\s*[=;]"

# Rust let bindings
rg "^\s*let (mut )?\w+"

# Constants in various languages
rg "^const \w+"
rg "^(static )?final \w+"
rg "^#define \w+"
```

### Comments and Documentation

```bash
# Single-line comments (C-style)
rg "//.*"

# Multi-line comments (C-style)
rg -U "/\*.*?\*/"

# Python docstrings
rg '""".*?"""'
rg -U '""".*?"""'

# TODO/FIXME/HACK comments
rg "TODO:"
rg "FIXME:"
rg "HACK:"
rg "XXX:"
rg "NOTE:"
rg "(TODO|FIXME|HACK|XXX|NOTE):"

# JSDoc comments
rg -U "/\*\*.*?\*/"

# Python comments
rg "^\s*#.*"
```

### Error Handling

```bash
# Try-catch blocks
rg "try\s*\{"
rg "catch\s*\("
rg "except\s+.*:"

# Error returns (Go)
rg "if err != nil"
rg "return.*err"

# Raise/throw statements
rg "raise \w+"
rg "throw new \w+"

# Error logging
rg "log\.error"
rg "console\.error"
rg "logger\.error"
```

### API Keys and Secrets

```bash
# Generic API keys
rg -i "api[_-]?key"
rg -i "api[_-]?secret"
rg -i "apikey\s*=\s*['\"]?\w+"

# AWS credentials
rg "AKIA[0-9A-Z]{16}"              # AWS Access Key ID
rg "aws_secret_access_key"

# GitHub tokens
rg "ghp_[0-9a-zA-Z]{36}"           # GitHub Personal Access Token
rg "gho_[0-9a-zA-Z]{36}"           # GitHub OAuth Token

# Slack tokens
rg "xox[baprs]-[0-9a-zA-Z-]+"

# Generic secrets
rg -i "(password|passwd|pwd)\s*=\s*['\"]?\w+"
rg -i "secret\s*=\s*['\"]?\w+"
rg -i "token\s*=\s*['\"]?\w+"

# Private keys
rg "BEGIN.*PRIVATE KEY"
rg "-----BEGIN RSA PRIVATE KEY-----"

# Database connection strings
rg "mongodb://.*@"
rg "mysql://.*@"
rg "postgres://.*@"
```

### Test Files and Test Cases

```bash
# Python tests
rg "def test_\w+"
rg "class Test\w+"

# JavaScript tests
rg "describe\(['\"]"
rg "it\(['\"]"
rg "test\(['\"]"

# Go tests
rg "func Test\w+\(t \*testing\.T\)"

# Rust tests
rg "#\[test\]"

# Ruby tests
rg "def test_\w+"
rg "it ['\"].*['\"] do"
```

## Output Formatting

### Color and Styling

```bash
# Force color output
rg --color always "pattern"
rg --color always "pattern" | less -R

# Disable colors
rg --color never "pattern"

# Auto color (default, colors for TTY)
rg --color auto "pattern"

# Custom colors
rg --colors 'match:fg:red' --colors 'path:fg:blue' "pattern"

# Available color types:
# - path: file path
# - line: line numbers
# - column: column numbers
# - match: matched text

# Available color specs:
# - fg:color (foreground color)
# - bg:color (background color)
# - style:bold, intense, underline
```

### Output Formats

```bash
# Default format (show filename, line number, content)
rg "pattern"

# Compact format (no line numbers, no colors)
rg -N --color never "pattern"

# Machine-readable format (null-separated)
rg --null "pattern"

# JSON output
rg --json "pattern"

# Vim-style output (filename:line:column:content)
rg --vimgrep "pattern"

# Custom path separator
rg --path-separator '/' "pattern"

# Custom heading
rg --heading "pattern"           # Group by file (default with TTY)
rg --no-heading "pattern"        # Don't group by file

# Show file paths as hyperlinks (some terminals)
rg --hyperlink-format default "pattern"
```

### Statistics and Summary

```bash
# Count matches per file
rg -c "pattern"

# Count total matches (not lines)
rg --count-matches "pattern"

# Show statistics
rg --stats "pattern"

# Quiet mode (only exit code)
rg -q "pattern"

# Only show filenames with matches
rg -l "pattern"

# Only show filenames without matches
rg --files-without-match "pattern"
```

### Limiting Output

```bash
# Limit matches per file
rg -m 5 "pattern"
rg --max-count 5 "pattern"

# Stop after first match
rg -m 1 "pattern"

# Limit total number of results
# (no direct option, use head)
rg "pattern" | head -n 20
```

## File Listing and Filtering

### List Files

```bash
# List all files that would be searched
rg --files

# List files of specific type
rg --files -t py
rg --files --type rust

# List files matching glob
rg --files -g "*.js"
rg --files --glob "**/*.py"

# List files with specific encoding
rg --files -E utf8
```

### Type Definitions

```bash
# Show all type definitions
rg --type-list

# Custom type definition
rg --type-add 'web:*.{html,css,js}' -t web "pattern"

# Multiple patterns in custom type
rg --type-add 'config:*.{yml,yaml,json,toml}' -t config "pattern"

# Add type for this session
rg --type-add 'custom:*.foo' -t custom "pattern"
```

### Ignore Files

```bash
# Use .gitignore (default)
rg "pattern"

# Also use .ignore files
rg "pattern"

# Use .rgignore files
rg "pattern"

# Ignore specific patterns
rg --ignore-file custom-ignore.txt "pattern"

# Don't use ignore files
rg --no-ignore "pattern"

# Don't use .gitignore
rg --no-ignore-vcs "pattern"

# Don't use global ignore files
rg --no-ignore-global "pattern"
```

### Custom Ignore Patterns

Create `.rgignore` file:
```
# .rgignore example
*.log
*.tmp
node_modules/
.git/
dist/
build/
__pycache__/
*.pyc
.DS_Store
```

## Performance Optimization

### Parallel Search

```bash
# Default: automatic parallelism based on CPU cores
rg "pattern"

# Specify number of threads
rg -j 4 "pattern"
rg --threads 4 "pattern"

# Single-threaded search
rg -j 1 "pattern"

# Maximum threads
rg -j $(nproc) "pattern"
```

### Memory Management

```bash
# Use memory-mapped files (faster for large files)
rg --mmap "pattern"

# Don't use memory mapping
rg --no-mmap "pattern"

# Auto (default, uses heuristics)
# Uses mmap for large files, regular reading for small files
```

### Optimizing Searches

```bash
# Use fixed-strings for literal matches (faster)
rg -F "literal_string"

# Limit file types to reduce search space
rg -t py "pattern"

# Use more specific patterns
rg "^import specific" vs rg "import"

# Limit recursion depth
rg --max-depth 2 "pattern"

# Skip large files
rg --max-filesize 1M "pattern"

# Combine multiple optimizations
rg -F -t py --max-depth 3 "literal_pattern"
```

### Benchmarking

```bash
# Time the search
time rg "pattern"

# With stats
rg --stats "pattern"

# Compare different options
time rg "pattern"
time rg -F "pattern"
time rg -t py "pattern"
```

## Practical Use Cases

### Code Refactoring

```bash
# Find all usages of a function
rg "\bfunction_name\b"
rg -w "function_name"

# Find all usages with context
rg -C 3 "function_name"

# Show which files use a function
rg -l "function_name"

# Count usages per file
rg -c "function_name"

# Find and show replacements
rg "oldName" -r "newName"

# Find function definitions and calls
rg "def function_name|function_name\("
```

### Security Auditing

```bash
# Find potential secrets
rg -i "(password|secret|api_key|token)\s*=\s*['\"]?\w+"

# Find TODO/FIXME in security contexts
rg "TODO.*security"
rg "FIXME.*(auth|password|token)"

# Find SQL queries (potential injection points)
rg "SELECT.*FROM"
rg "execute\(.*SELECT"

# Find eval/exec (potential code injection)
rg "\beval\("
rg "\bexec\("

# Find file operations
rg "open\(['\"]"
rg "readFile|writeFile"

# Find network operations
rg "http\.request"
rg "fetch\("
rg "requests\.get|requests\.post"
```

### Log Analysis

```bash
# Find errors in logs
rg "ERROR|FATAL|CRITICAL" logs/

# Find errors with timestamp
rg "\d{4}-\d{2}-\d{2}.*ERROR" logs/

# Find specific error codes
rg "HTTP [45]\d{2}"
rg "status.*[45]\d{2}"

# Find exceptions
rg "Exception|Traceback" logs/

# Search compressed logs
rg -z "ERROR" logs/*.gz

# Find slow queries
rg "duration.*[0-9]{4,}" logs/

# Search logs by date
rg "2024-01-15" logs/
rg "Jan 15" logs/
```

### Documentation Search

```bash
# Search markdown files
rg -t md "pattern"

# Search in code comments and docs
rg "//.*pattern|/\*.*pattern" -t cpp
rg "#.*pattern" -t py

# Find specific sections
rg "^## \w+" -t md

# Find TODO items in docs
rg "TODO" -t md

# Search across multiple doc formats
rg -t md -t rst -t txt "pattern"
```

### Configuration Management

```bash
# Search config files
rg -t yaml -t json -t toml "pattern"

# Find specific settings
rg "debug\s*=\s*true" -t yaml

# Find database configs
rg "host.*:.*port" -t yaml

# Find environment variables
rg "\$\{?\w+\}?" -g "*.env"

# Search INI files
rg --type-add 'ini:*.ini' -t ini "pattern"
```

### Finding Duplicated Code

```bash
# Find similar function signatures
rg "def \w+\([^)]*\):" -t py | sort | uniq -c

# Find repeated patterns
rg "console\.log" | wc -l

# Find copied error messages
rg "error occurred" -c
```

### Dependency Analysis

```bash
# Find all imports of a module
rg "import.*module_name"
rg "from module_name import"

# Find package versions
rg "==\d+\.\d+\.\d+" requirements.txt
rg "\"version\":\s*\"" package.json

# Find outdated copyright years
rg "Copyright.*201[0-9]"

# Find specific library usage
rg "import requests"
rg "import.*pandas"
```

### Build and CI/CD

```bash
# Find failing tests
rg "FAILED|ERROR" test-results/

# Find deprecated warnings
rg "DeprecationWarning"
rg "deprecated"

# Check for hardcoded values
rg "localhost:3000"
rg "http://127\.0\.0\.1"

# Find debug code
rg "console\.log"
rg "debugger"
rg "import pdb"
```

## Integration with Other Tools

### With Git

```bash
# Search only tracked files
git ls-files | rg "pattern"

# Search files changed in last commit
git diff --name-only HEAD~1 | xargs rg "pattern"

# Search in specific branch
git show branch:file.txt | rg "pattern"

# Search commit messages
git log --all --grep="pattern"

# Combine with git grep
git ls-files | rg "pattern"
```

### With Find

```bash
# ripgrep replaces most find use cases
# But you can combine them:

# Find files, then search content
find . -name "*.py" | xargs rg "pattern"

# Better: use ripgrep's built-in filtering
rg -t py "pattern"

# Find files modified in last day, then search
find . -mtime -1 -type f | xargs rg "pattern"
```

### With Sed/Awk

```bash
# ripgrep to find, sed to replace
rg -l "oldtext" | xargs sed -i 's/oldtext/newtext/g'

# ripgrep with awk for processing
rg "pattern" | awk '{print $1}'

# Extract specific fields
rg "error" logs/ | awk -F: '{print $1}' | sort | uniq
```

### With Vim

```bash
# Use ripgrep as grep program in Vim
# Add to .vimrc:
# set grepprg=rg\ --vimgrep\ --no-heading\ --smart-case
# set grepformat=%f:%l:%c:%m

# Then in Vim:
:grep pattern
:copen

# Use with fzf.vim
# :Rg pattern
```

### With FZF

```bash
# Interactive file search
rg --files | fzf

# Interactive content search
rg --no-heading --color always "pattern" | fzf --ansi

# Live search
fzf --preview 'rg --pretty --context 3 {q} || true' --phony -q ""

# Bash key binding
# Add to .bashrc:
# bind '"\C-f": "rg --files | fzf\n"'
```

### With Clipboard (xclip/pbcopy)

```bash
# Copy results to clipboard (Linux)
rg "pattern" | xclip -selection clipboard

# Copy results to clipboard (macOS)
rg "pattern" | pbcopy

# Copy just filenames
rg -l "pattern" | xclip -selection clipboard
```

### With Watch

```bash
# Monitor changes in real-time
watch -n 1 'rg "ERROR" logs/latest.log | tail -20'

# Watch for new matches
watch -n 2 'rg -c "pattern"'
```

### With Entr

```bash
# Re-run search when files change
rg --files | entr -c rg "pattern"

# Run tests when files change
rg --files -t py | entr -c pytest
```

### Pipes and Filters

```bash
# Count unique matches
rg -o "pattern" | sort | uniq | wc -l

# Most common matches
rg -o "\b\w+\b" | sort | uniq -c | sort -rn | head -20

# Filter results
rg "pattern" | grep -v "exclude_this"

# Format output
rg "pattern" | column -t

# Extract and process
rg -o "email@\w+\.\w+" | sort -u > emails.txt
```

## Scripting with ripgrep

### Bash Scripts

```bash
#!/bin/bash
# Find and count TODOs by author

echo "TODO count by file:"
rg -c "TODO" | while IFS=: read -r file count; do
    if [ "$count" -gt 0 ]; then
        printf "%-50s %d\n" "$file" "$count"
    fi
done | sort -k2 -rn
```

```bash
#!/bin/bash
# Security audit script

echo "=== Potential Security Issues ==="

echo -e "\n[*] Looking for hardcoded passwords..."
rg -i "password\s*=\s*['\"][^'\"]+['\"]" -g '!*.{md,txt}'

echo -e "\n[*] Looking for API keys..."
rg -i "api[_-]?key\s*=\s*['\"][^'\"]+['\"]"

echo -e "\n[*] Looking for AWS credentials..."
rg "AKIA[0-9A-Z]{16}"

echo -e "\n[*] Looking for private keys..."
rg "BEGIN.*PRIVATE KEY"

echo -e "\n[*] Looking for potential SQL injection..."
rg "execute.*SELECT.*\+" -t py

echo -e "\n[*] Looking for eval/exec usage..."
rg "\b(eval|exec)\(" -t py
```

```bash
#!/bin/bash
# Find and replace across files

PATTERN="$1"
REPLACEMENT="$2"

if [ -z "$PATTERN" ] || [ -z "$REPLACEMENT" ]; then
    echo "Usage: $0 <pattern> <replacement>"
    exit 1
fi

# Show preview
echo "Preview of changes:"
rg "$PATTERN" -r "$REPLACEMENT" --color always | head -20

read -p "Continue with replacement? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Get list of files
    FILES=$(rg -l "$PATTERN")

    # Replace in each file
    for file in $FILES; do
        echo "Processing $file..."
        sed -i "s/$PATTERN/$REPLACEMENT/g" "$file"
    done

    echo "Done!"
fi
```

### Python Integration

```python
#!/usr/bin/env python3
import subprocess
import json
import sys

def search_with_rg(pattern, path='.', file_type=None):
    """Search using ripgrep and return results as structured data."""
    cmd = ['rg', '--json', pattern]

    if file_type:
        cmd.extend(['-t', file_type])

    cmd.append(path)

    result = subprocess.run(cmd, capture_output=True, text=True)

    matches = []
    for line in result.stdout.strip().split('\n'):
        if line:
            data = json.loads(line)
            if data.get('type') == 'match':
                matches.append({
                    'file': data['data']['path']['text'],
                    'line_number': data['data']['line_number'],
                    'line': data['data']['lines']['text'].strip(),
                })

    return matches

def find_todos():
    """Find all TODO items and organize by file."""
    matches = search_with_rg('TODO:')

    by_file = {}
    for match in matches:
        file = match['file']
        if file not in by_file:
            by_file[file] = []
        by_file[file].append(match)

    for file, items in sorted(by_file.items()):
        print(f"\n{file}:")
        for item in items:
            print(f"  Line {item['line_number']}: {item['line']}")

def analyze_imports(file_type='py'):
    """Analyze import statements in codebase."""
    pattern = '^(import|from) .*'
    matches = search_with_rg(pattern, file_type=file_type)

    imports = {}
    for match in matches:
        module = match['line'].split()[1]
        imports[module] = imports.get(module, 0) + 1

    print("Most used imports:")
    for module, count in sorted(imports.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {module}: {count}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        results = search_with_rg(pattern)
        print(f"Found {len(results)} matches")
        for r in results[:10]:
            print(f"{r['file']}:{r['line_number']}: {r['line']}")
    else:
        find_todos()
```

### JSON Output Processing

```bash
# Parse JSON output with jq
rg --json "pattern" | jq -s '[.[] | select(.type == "match") | .data.path.text] | unique'

# Extract specific fields
rg --json "error" | jq -r 'select(.type == "match") | "\(.data.path.text):\(.data.line_number)"'

# Count matches per file
rg --json "pattern" | jq -s 'group_by(.data.path.text) | map({file: .[0].data.path.text, count: length})'

# Get statistics
rg --json "pattern" | jq -s 'length'
```

## Comparison with Other Tools

### ripgrep vs grep

| Feature | ripgrep | grep |
|---------|---------|------|
| Speed | Much faster (2-10x) | Slower |
| Recursive search | Default | Need -r flag |
| Gitignore support | Yes (default) | No |
| Unicode support | Full | Limited |
| PCRE support | Yes (-P) | Depends on version |
| Binary file handling | Smart (auto-skip) | Basic |
| Parallel search | Yes | No |
| Compressed files | Yes (-z) | Need zgrep |

```bash
# ripgrep equivalent of common grep commands

# grep -r "pattern" .
rg "pattern"

# grep -i "pattern" file
rg -i "pattern" file

# grep -w "word" file
rg -w "word" file

# grep -v "pattern" file
rg --invert-match "pattern" file

# grep -l "pattern" *
rg -l "pattern"

# grep -c "pattern" file
rg -c "pattern" file

# grep -A 3 -B 3 "pattern" file
rg -C 3 "pattern" file

# grep -E "pattern1|pattern2" file
rg "pattern1|pattern2" file
```

### ripgrep vs ag (The Silver Searcher)

| Feature | ripgrep | ag |
|---------|---------|-----|
| Speed | Faster | Fast |
| Regex engine | Rust regex / PCRE2 | PCRE |
| Memory usage | Lower | Higher |
| Active development | Very active | Less active |
| Compressed files | Yes | No |
| Encoding support | Excellent | Good |

```bash
# ag equivalent commands

# ag "pattern"
rg "pattern"

# ag -l "pattern"
rg -l "pattern"

# ag -i "pattern"
rg -i "pattern"

# ag --ignore-dir dir "pattern"
rg -g '!dir/**' "pattern"
```

### ripgrep vs ack

| Feature | ripgrep | ack |
|---------|---------|-----|
| Speed | Much faster | Slower |
| Language | Rust | Perl |
| Dependencies | None (binary) | Perl required |
| File type detection | Excellent | Excellent |
| Customization | Good | Excellent |

## Configuration

### Config File

Create `~/.ripgreprc` or set `RIPGREP_CONFIG_PATH`:

```bash
# ~/.ripgreprc example

# Smart case searching
--smart-case

# Show column numbers
--column

# Search hidden files
--hidden

# Don't search these directories
--glob=!.git/
--glob=!node_modules/
--glob=!.venv/
--glob=!__pycache__/
--glob=!*.min.js
--glob=!*.map

# Custom file types
--type-add=web:*.{html,css,js,jsx,tsx}
--type-add=config:*.{yaml,yml,json,toml,ini}

# Default colors
--colors=line:fg:yellow
--colors=match:fg:red
--colors=match:style:bold
```

Use the config file:
```bash
# Set environment variable
export RIPGREP_CONFIG_PATH="$HOME/.ripgreprc"

# Add to .bashrc or .zshrc
echo 'export RIPGREP_CONFIG_PATH="$HOME/.ripgreprc"' >> ~/.bashrc
```

### Shell Aliases

```bash
# Add to .bashrc or .zshrc

# Common searches
alias rgf='rg --files | rg'              # Search filenames
alias rgi='rg -i'                         # Case insensitive
alias rgl='rg -l'                         # List files only
alias rgc='rg -C 3'                       # With context
alias rgt='rg -t'                         # By file type

# Code search aliases
alias rgpy='rg -t py'                     # Python files
alias rgjs='rg -t js'                     # JavaScript files
alias rggo='rg -t go'                     # Go files
alias rgrs='rg -t rust'                   # Rust files

# Security audit aliases
alias rgsec='rg -i "(password|secret|api_key|token)\s*=\s*"'
alias rgaws='rg "AKIA[0-9A-Z]{16}"'

# Development aliases
alias rgtodo='rg "TODO|FIXME|HACK|XXX|NOTE"'
alias rgbug='rg -i "bug|issue|problem"'
alias rgtest='rg -t py "def test_"'

# Git-related
alias rgstaged='git diff --staged --name-only | xargs rg'
alias rgchanged='git diff --name-only | xargs rg'
```

### Environment Variables

```bash
# Config file location
export RIPGREP_CONFIG_PATH="$HOME/.ripgreprc"

# Set default options
export RIPGREP_ARGS="--smart-case --hidden"

# Disable colors (useful for scripts)
export RIPGREP_COLOR=never

# Custom type definitions
# (better to put in config file)
```

## Advanced Techniques

### Preprocessing

```bash
# Use preprocessor for custom file handling
# Create a preprocessor script

# Example: decompress before search
rg --pre gunzip --pre-glob '*.gz' "pattern"

# Example: search in PDFs (requires pdftotext)
rg --pre pdftotext --pre-glob '*.pdf' "pattern"

# Example: search in Office docs (requires catdoc, etc.)
#!/bin/bash
# preprocessor.sh
case "$1" in
    *.doc) catdoc "$1" ;;
    *.docx) docx2txt "$1" - ;;
    *.pdf) pdftotext "$1" - ;;
    *.odt) odt2txt "$1" ;;
    *) cat "$1" ;;
esac

# Use it
rg --pre ./preprocessor.sh --pre-glob '*.{doc,docx,pdf,odt}' "pattern"
```

### Complex Filters

```bash
# Multiple type filters
rg -t py -t js "pattern"

# Multiple glob patterns
rg -g '*.{py,pyx,pxd}' "pattern"

# Exclude multiple patterns
rg -g '!*.min.{js,css}' -g '!vendor/**' "pattern"

# Combine type and glob
rg -t py -g '!test_*' "pattern"

# Complex boolean filters using multiple invocations
rg "pattern" | rg -v "exclude" | rg "include"
```

### Working with Encodings

```bash
# Specify encoding
rg -E utf8 "pattern"
rg -E latin1 "pattern"
rg -E utf16le "pattern"

# Auto-detect encoding (default)
rg "pattern"

# Search across multiple encodings
for enc in utf8 latin1 utf16le; do
    echo "=== $enc ==="
    rg -E $enc "pattern"
done
```

### Negation and Inversion

```bash
# Invert match (show lines NOT matching)
rg -v "pattern"
rg --invert-match "pattern"

# Files without matches
rg --files-without-match "pattern"

# Exclude file types
rg -T js "pattern"

# Exclude paths
rg -g '!vendor/**' "pattern"

# Show context but exclude matches
rg -v "exclude" | rg -C 2 "include"
```

### Sorting and Uniqueness

```bash
# Sort files by name
rg "pattern" | sort

# Sort by line number
rg "pattern" | sort -t: -k2 -n

# Unique matches only
rg -o "pattern" | sort -u

# Count unique matches
rg -o "pattern" | sort | uniq | wc -l

# Most common matches
rg -o "\b\w+\b" | sort | uniq -c | sort -rn | head -20
```

## Troubleshooting

### Common Issues

```bash
# Pattern not found but should be
# Check: hidden files, gitignore, file types

# Include hidden files
rg --hidden "pattern"

# Ignore gitignore
rg --no-ignore "pattern"

# Search all files (including binary)
rg -uuu "pattern"

# Check what files would be searched
rg --files

# Regex not matching
# Try literal search
rg -F "literal.string"

# Try different regex engine
rg -P "pcre2_pattern"

# Performance issues
# Use fixed strings if possible
rg -F "literal"

# Limit file types
rg -t py "pattern"

# Reduce threads if CPU-bound
rg -j 2 "pattern"

# Encoding issues
# Try different encodings
rg -E latin1 "pattern"
rg -E utf16le "pattern"

# Include binary files
rg -a "pattern"
```

### Debug Mode

```bash
# Show debug information
rg --debug "pattern" 2>&1 | less

# Trace what's being searched
rg --debug "pattern" 2>&1 | grep "search path"

# Check ignore rules
rg --debug "pattern" 2>&1 | grep -i "ignore"

# Verify regex compilation
rg --debug "pattern" 2>&1 | grep -i "regex"
```

### Permission Errors

```bash
# Permission denied errors
# Use sudo if needed
sudo rg "pattern" /root/

# Skip permission errors
rg "pattern" 2>/dev/null

# Show only permission errors
rg "pattern" 2>&1 >/dev/null
```

## Best Practices

### General Guidelines

1. **Use specific patterns** - More specific patterns are faster
   ```bash
   rg "^import specific" # Better than rg "import"
   ```

2. **Use file type filters** - Reduce search space
   ```bash
   rg -t py "pattern" # Better than rg "pattern"
   ```

3. **Use literal search when possible** - Faster than regex
   ```bash
   rg -F "literal.string" # Better than rg "literal\.string"
   ```

4. **Leverage gitignore** - Default behavior is usually right
   ```bash
   rg "pattern" # Respects .gitignore
   ```

5. **Use word boundaries for identifiers**
   ```bash
   rg -w "function_name" # Better than rg "function_name"
   ```

### Performance Tips

1. **Start with narrow searches**
   ```bash
   rg -t py --max-depth 2 "specific_pattern"
   ```

2. **Use appropriate thread count**
   ```bash
   rg -j 4 "pattern" # For CPU-bound systems
   ```

3. **Avoid unnecessary context**
   ```bash
   rg "pattern" # Instead of rg -C 10 "pattern" when not needed
   ```

4. **Use count when possible**
   ```bash
   rg -c "pattern" # Faster than rg "pattern" | wc -l
   ```

5. **Limit output early**
   ```bash
   rg -m 10 "pattern" # Stop after 10 matches per file
   ```

### Security Best Practices

1. **Regular secret scanning**
   ```bash
   rg -i "(password|secret|api_key|token)\s*=\s*['\"]?\w+"
   ```

2. **Check before committing**
   ```bash
   # Add to git pre-commit hook
   rg --quiet "FIXME|TODO|password.*=" && exit 1
   ```

3. **Audit dependencies**
   ```bash
   rg "==|>=|<=" requirements.txt
   rg "\"version\":" package.json
   ```

4. **Find debug code**
   ```bash
   rg "console\.(log|debug)|debugger|import pdb"
   ```

### Code Quality

1. **Find TODOs regularly**
   ```bash
   rg "TODO|FIXME|HACK|XXX" -g '!vendor/**'
   ```

2. **Check for code smells**
   ```bash
   rg "eval\(|exec\(" -t py
   rg "var " -t js
   ```

3. **Monitor test coverage**
   ```bash
   rg "def test_" -t py -c
   ```

4. **Find duplicated code**
   ```bash
   rg "error message" -c | sort -t: -k2 -rn
   ```

## Quick Reference

### Essential Commands

```bash
# Basic search
rg "pattern"

# Case-insensitive
rg -i "pattern"

# Whole words only
rg -w "word"

# Fixed strings (literal)
rg -F "literal.string"

# File type filter
rg -t py "pattern"

# Exclude file type
rg -T js "pattern"

# Show filenames only
rg -l "pattern"

# Count matches
rg -c "pattern"

# With context
rg -C 3 "pattern"

# Include hidden files
rg --hidden "pattern"

# Ignore .gitignore
rg --no-ignore "pattern"

# List files
rg --files

# Multiline search
rg -U "pattern.*\n.*pattern"

# Replace preview
rg "old" -r "new"

# JSON output
rg --json "pattern"

# Statistics
rg --stats "pattern"
```

### Common Options

| Option | Short | Description |
|--------|-------|-------------|
| `--type` | `-t` | Filter by file type |
| `--type-not` | `-T` | Exclude file type |
| `--glob` | `-g` | Include/exclude by glob |
| `--files` | | List files to search |
| `--ignore-case` | `-i` | Case-insensitive search |
| `--smart-case` | `-S` | Smart case (default) |
| `--word-regexp` | `-w` | Match whole words |
| `--fixed-strings` | `-F` | Literal strings |
| `--count` | `-c` | Count matches per file |
| `--files-with-matches` | `-l` | Show filenames only |
| `--context` | `-C` | Show context lines |
| `--after-context` | `-A` | Show lines after |
| `--before-context` | `-B` | Show lines before |
| `--only-matching` | `-o` | Show only matched part |
| `--multiline` | `-U` | Multiline search |
| `--replace` | `-r` | Show replacement |
| `--hidden` | | Search hidden files |
| `--no-ignore` | | Don't respect ignore files |
| `--max-depth` | | Limit recursion depth |

### File Type Shortcuts

```bash
rg -t py      # Python
rg -t js      # JavaScript
rg -t ts      # TypeScript
rg -t rust    # Rust
rg -t go      # Go
rg -t cpp     # C++
rg -t c       # C
rg -t java    # Java
rg -t ruby    # Ruby
rg -t php     # PHP
rg -t html    # HTML
rg -t css     # CSS
rg -t md      # Markdown
rg -t json    # JSON
rg -t yaml    # YAML
rg -t xml     # XML
rg -t sql     # SQL
rg -t sh      # Shell scripts
```

### Common Patterns

```bash
# Functions
rg "^(def|function|func|fn) \w+"

# Classes
rg "^(class|struct|interface) \w+"

# Imports
rg "^(import|from|#include|use) "

# TODOs
rg "TODO|FIXME|HACK|XXX|NOTE"

# Email
rg "\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

# IP addresses
rg "\b(?:\d{1,3}\.){3}\d{1,3}\b"

# URLs
rg "https?://[^\s]+"

# Hex colors
rg "#[0-9a-fA-F]{6}\b"

# UUIDs
rg "[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
```

## Conclusion

ripgrep is a modern, fast, and user-friendly search tool that has become essential for developers. Its intelligent defaults, respect for version control systems, and excellent performance make it the go-to choice for code searching.

**Key Takeaways:**
- ripgrep is significantly faster than traditional grep and alternatives
- Smart defaults (gitignore, smart case, recursive) make it intuitive
- Extensive file type support and filtering options
- Powerful regex support with both Rust regex and PCRE2 engines
- Excellent Unicode and encoding support
- Integrates well with editors, shells, and other tools
- Highly configurable through config files and environment variables

**Learning Path:**
1. **Day 1**: Basic searches, file type filtering, simple patterns
2. **Week 1**: Context, output formatting, glob patterns, common use cases
3. **Week 2**: Advanced regex, multiline search, replacement preview
4. **Month 1**: Configuration, shell integration, scripting
5. **Month 2+**: Advanced filtering, preprocessing, performance optimization

**Resources:**
- Official repository: https://github.com/BurntSushi/ripgrep
- User guide: https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md
- Regex syntax: https://docs.rs/regex/latest/regex/#syntax
- Configuration: https://github.com/BurntSushi/ripgrep/blob/master/GUIDE.md#configuration-file

**When to Use ripgrep:**
- Searching large codebases
- Quick file content exploration
- Code refactoring and analysis
- Log file analysis
- Security auditing
- Any recursive text search

**When to Use Alternatives:**
- Need POSIX compliance (use grep)
- Embedded systems without ripgrep (use grep)
- Very specific edge cases requiring special grep features
- Already have complex grep scripts (though ripgrep is usually compatible)

ripgrep's combination of speed, usability, and intelligent defaults makes it an indispensable tool for modern development workflows. Once you start using it, you'll wonder how you lived without it.

Happy searching!
