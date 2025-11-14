# sed

sed (Stream Editor) is a powerful text processing utility that performs editing operations on text streams and files. It reads input line by line, applies commands, and outputs the result. sed is particularly useful for automated text transformations, search and replace operations, and text filtering.

## Overview

sed is a non-interactive editor that processes text one line at a time using a simple programming language. It's especially powerful in shell scripts and command pipelines for text manipulation tasks.

**Key Features:**
- Line-by-line stream processing
- Regular expression support
- In-place file editing
- Pattern space and hold space for complex operations
- Branching and looping capabilities
- Minimal memory footprint
- POSIX standard compatibility

**Common Use Cases:**
- Search and replace operations
- Text deletion and insertion
- Line filtering and selection
- Text transformation and formatting
- Configuration file manipulation
- Log file processing

## Basic Syntax

```bash
# General syntax
sed [options] 'command' file
sed [options] -e 'command1' -e 'command2' file
sed [options] -f script.sed file

# Common options
sed -n 'command' file      # Suppress automatic output
sed -i 'command' file      # Edit file in-place
sed -i.bak 'command' file  # Edit in-place with backup
sed -e 'cmd1' -e 'cmd2'    # Multiple commands
```

## Addressing

Addresses specify which lines a command applies to. You can use line numbers, patterns, or ranges.

### Line Number Addressing

```bash
# Single line
sed '5d' file              # Delete line 5
sed '3p' file              # Print line 3 (plus all lines)
sed -n '3p' file           # Print only line 3

# Last line
sed '$d' file              # Delete last line
sed -n '$p' file           # Print last line

# Range of lines
sed '2,5d' file            # Delete lines 2-5
sed '10,20p' file          # Print lines 10-20
sed '1,10s/old/new/g' file # Replace in lines 1-10

# From line to end
sed '5,$d' file            # Delete from line 5 to end

# Every nth line
sed -n '1~2p' file         # Print odd lines (1, 3, 5, ...)
sed -n '2~2p' file         # Print even lines (2, 4, 6, ...)
sed '0~5d' file            # Delete every 5th line
```

### Pattern Addressing

```bash
# Single pattern
sed '/pattern/d' file              # Delete lines matching pattern
sed '/ERROR/p' file                # Print lines containing ERROR
sed -n '/^#/p' file                # Print comment lines

# Pattern range
sed '/start/,/end/d' file          # Delete from start to end pattern
sed '/BEGIN/,/END/p' file          # Print lines between patterns
sed '/^$/,/^$/d' file              # Delete blank line groups

# Negation
sed '/pattern/!d' file             # Delete lines NOT matching pattern
sed -n '/pattern/!p' file          # Print lines NOT matching pattern

# Multiple patterns
sed '/pattern1/d; /pattern2/d' file  # Delete lines matching either pattern
```

### Advanced Addressing

```bash
# Line number and pattern
sed '5,/pattern/d' file            # Delete from line 5 to pattern match
sed '/pattern/,10d' file           # Delete from pattern to line 10

# Step addressing
sed -n '1~3p' file                 # Print every 3rd line starting from 1
sed '2~4d' file                    # Delete every 4th line starting from 2

# Address with offset
sed '/pattern/,+5d' file           # Delete match and 5 lines after
sed '10,+3d' file                  # Delete lines 10-13
```

## Basic Commands

### Substitute Command (s)

The most commonly used sed command for search and replace.

```bash
# Basic substitution
sed 's/old/new/' file              # Replace first occurrence per line
sed 's/old/new/g' file             # Replace all occurrences
sed 's/old/new/2' file             # Replace second occurrence
sed 's/old/new/2g' file            # Replace from second occurrence onward

# Case-insensitive substitution
sed 's/old/new/i' file             # Case-insensitive replacement
sed 's/old/new/gi' file            # Case-insensitive, all occurrences

# Print only changed lines
sed -n 's/old/new/p' file          # Print only lines where substitution occurred

# Write changes to file
sed -n 's/old/new/w output.txt' file  # Write changed lines to file
```

### Delete Command (d)

```bash
# Delete specific lines
sed '5d' file                      # Delete line 5
sed '1,3d' file                    # Delete lines 1-3
sed '$d' file                      # Delete last line

# Delete by pattern
sed '/pattern/d' file              # Delete lines matching pattern
sed '/^$/d' file                   # Delete empty lines
sed '/^#/d' file                   # Delete comment lines
sed '/^\s*$/d' file                # Delete blank lines (with whitespace)

# Delete ranges
sed '/start/,/end/d' file          # Delete from start to end pattern
```

### Print Command (p)

```bash
# Print specific lines (use with -n)
sed -n '5p' file                   # Print line 5 only
sed -n '1,10p' file                # Print lines 1-10
sed -n '$p' file                   # Print last line

# Print by pattern
sed -n '/pattern/p' file           # Print matching lines
sed -n '/ERROR/p' logfile          # Print error lines
sed -n '/^[0-9]/p' file            # Print lines starting with digit

# Print with duplicates
sed '5p' file                      # Print line 5 twice (line + print)
```

### Quit Command (q)

```bash
# Quit after line number
sed '10q' file                     # Print first 10 lines and quit
sed -n '1,20p; 20q' file          # Print lines 1-20 then quit

# Quit after pattern
sed '/pattern/q' file              # Print up to first match and quit
sed '/ERROR/q' file                # Stop at first ERROR
```

### Transform Command (y)

```bash
# Character-by-character replacement
sed 'y/abc/ABC/' file              # Replace a→A, b→B, c→C
sed 'y/abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ/' file  # To uppercase
sed 'y/0123456789/----------/' file  # Replace digits with dashes
```

## Substitution in Detail

### Regular Expressions

```bash
# Anchors
sed 's/^#//' file                  # Remove # from line start
sed 's/;$//' file                  # Remove ; from line end
sed 's/^/> /' file                 # Add > to line start
sed 's/$/ END/' file               # Add END to line end

# Character classes
sed 's/[0-9]/#/g' file             # Replace digits with #
sed 's/[a-z]/*/g' file             # Replace lowercase with *
sed 's/[[:space:]]/_/g' file       # Replace whitespace with _
sed 's/[[:punct:]]//g' file        # Remove punctuation

# Quantifiers
sed 's/a*/x/' file                 # Replace a* with x
sed 's/a\+/x/' file                # Replace one or more a with x
sed 's/a\{3\}/x/' file             # Replace exactly 3 a's with x
sed 's/a\{2,5\}/x/' file           # Replace 2-5 a's with x

# Groups and backreferences
sed 's/\(.*\)/[\1]/' file          # Wrap entire line in brackets
sed 's/\([0-9]*\)\.\([0-9]*\)/\2.\1/' file  # Swap decimal parts
sed 's/\(.*\):\(.*\)/\2:\1/' file  # Swap colon-separated parts
```

### Delimiters

```bash
# Alternative delimiters (useful for paths)
sed 's|/old/path|/new/path|g' file          # Using |
sed 's#/old/path#/new/path#g' file          # Using #
sed 's@/old/path@/new/path@g' file          # Using @
sed 's:/old/path:/new/path:g' file          # Using :

# Mixed delimiters
sed 's|http://|https://|g' file             # Convert HTTP to HTTPS
```

### Special Characters in Replacement

```bash
# Escaping special characters
sed 's/\$/DOLLAR/g' file           # Replace $ with DOLLAR
sed 's/\*/STAR/g' file             # Replace * with STAR
sed 's/\./DOT/g' file              # Replace . with DOT
sed 's/\//SLASH/g' file            # Replace / with SLASH

# Using matched pattern (&)
sed 's/[0-9]\+/(&)/g' file         # Wrap numbers in parentheses
sed 's/ERROR/*** & ***/g' file     # Wrap ERROR with asterisks
sed 's/^/Line: &/' file            # Prefix each line with "Line: "

# Newline in replacement
sed 's/,/,\n/g' file               # Replace comma with comma+newline
sed 's/;/;\n/g' file               # Split on semicolons
```

## Text Manipulation Commands

### Append (a)

```bash
# Append after line
sed '5a\New line text' file        # Append after line 5
sed '$a\Last line' file            # Append after last line

# Append after pattern
sed '/pattern/a\Added text' file   # Append after matching lines
sed '/^Section/a\---' file         # Add separator after sections

# Append multiple lines
sed '/pattern/a\
Line 1\
Line 2\
Line 3' file
```

### Insert (i)

```bash
# Insert before line
sed '1i\Header line' file          # Insert before first line
sed '5i\New line' file             # Insert before line 5

# Insert before pattern
sed '/pattern/i\Inserted text' file  # Insert before matching lines
sed '/^#/i\---' file               # Insert separator before comments

# Insert multiple lines
sed '1i\
#!/bin/bash\
# Script header\
' file
```

### Change (c)

```bash
# Replace entire line
sed '5c\New line content' file     # Replace line 5
sed '$c\New last line' file        # Replace last line

# Replace by pattern
sed '/pattern/c\Replacement line' file  # Replace matching lines
sed '/ERROR/c\[ERROR REDACTED]' file    # Replace error lines

# Replace range
sed '10,20c\--- SECTION REMOVED ---' file  # Replace lines 10-20 with one line
```

### Next (n) and Next+Print (N)

```bash
# Skip next line
sed 'n; s/pattern/replacement/' file   # Replace on every other line

# Read next line into pattern space
sed '/pattern/{N; s/\n/ /}' file       # Join lines after pattern match
```

## Pattern Space and Hold Space

sed maintains two buffers: pattern space (current line) and hold space (temporary storage).

### Hold Space Commands

```bash
# h - Copy pattern space to hold space
# H - Append pattern space to hold space
# g - Copy hold space to pattern space
# G - Append hold space to pattern space
# x - Exchange pattern and hold spaces

# Reverse file (using hold space)
sed -n '1!G; h; $p' file

# Print duplicate lines
sed -n '/pattern/{h; n; /pattern/{g; p}}' file

# Remove duplicate consecutive lines
sed '$!N; /^\(.*\)\n\1$/!P; D'
```

### Multi-line Operations

```bash
# Join lines with pattern
sed '/pattern/{N; s/\n/ /}' file       # Join pattern line with next

# Join all lines
sed ':a; N; $!ba; s/\n/ /g' file

# Join lines ending with backslash
sed -e ':a' -e '/\\$/N; s/\\\n//; ta' file

# Process paragraph at a time (blank line separated)
sed '/./{H;$!d;}; x; s/PATTERN/REPLACEMENT/g' file
```

## Advanced Features

### Labels and Branching

```bash
# Label syntax: :label
# Branch: b label (unconditional)
# Test: t label (branch if substitution succeeded)

# Remove duplicate consecutive lines
sed ':a; $!N; s/^\(.*\)\n\1$/\1/; ta; P; D' file

# Loop through replacements
sed ':a; s/pattern/replacement/; ta' file

# Conditional branching
sed '/pattern/b skip; s/old/new/; :skip' file
```

### Advanced Patterns

```bash
# Number lines
sed = file | sed 'N; s/\n/\t/'

# Number non-blank lines
sed '/./=' file | sed '/./N; s/\n/ /'

# Double space file
sed 'G' file

# Double space file (blank lines already present)
sed '/^$/d; G' file

# Triple space
sed 'G;G' file

# Reverse line order (tac alternative)
sed '1!G; h; $!d' file

# Reverse character order in each line
sed '/\n/!G; s/\(.\)\(.*\n\)/&\2\1/; //D; s/.//' file
```

## Common Patterns

### Find and Replace

```bash
# Simple replacement
sed 's/foo/bar/g' file                 # Replace all foo with bar
sed -i 's/foo/bar/g' file              # Replace in-place

# Multiple replacements
sed 's/foo/bar/g; s/baz/qux/g' file
sed -e 's/foo/bar/g' -e 's/baz/qux/g' file

# Replace only on specific lines
sed '10,20s/old/new/g' file            # Lines 10-20
sed '/pattern/s/old/new/g' file        # Lines matching pattern

# Replace whole words only
sed 's/\bword\b/replacement/g' file

# Replace with special characters
sed 's/$/\r/' file                     # Add Windows line endings
sed 's/\t/    /g' file                 # Replace tabs with spaces
```

### Line Deletion

```bash
# Delete empty lines
sed '/^$/d' file
sed '/^\s*$/d' file                    # Including whitespace-only lines

# Delete comment lines
sed '/^#/d' file                       # Shell/Python comments
sed '/^\/\//d' file                    # C++ style comments
sed '/^\/\*/,/\*\//d' file             # C style block comments

# Delete lines by pattern
sed '/pattern/d' file                  # Lines containing pattern
sed '/^pattern$/d' file                # Lines exactly matching
sed '/pattern1/d; /pattern2/d' file    # Multiple patterns

# Delete range
sed '10,20d' file                      # Delete lines 10-20
sed '/start/,/end/d' file              # Delete from start to end pattern
```

### Line Extraction

```bash
# Extract specific lines
sed -n '10p' file                      # Line 10
sed -n '10,20p' file                   # Lines 10-20
sed -n '1p; 5p; 10p' file              # Multiple specific lines

# Extract by pattern
sed -n '/pattern/p' file               # Lines matching pattern
sed -n '/start/,/end/p' file           # Lines between patterns
sed -n '/ERROR/p' file                 # Error lines

# Extract and modify
sed -n 's/.*pattern:\(.*\)/\1/p' file  # Extract after pattern:
sed -n 's/^.*=//p' file                # Extract after =
```

### Text Insertion and Formatting

```bash
# Add line numbers
sed = file | sed 'N; s/\n/\t/'

# Add prefix/suffix
sed 's/^/PREFIX: /' file               # Add prefix
sed 's/$/ [END]/' file                 # Add suffix
sed 's/.*/>>> & <<</' file             # Wrap lines

# Add header/footer
sed '1i\HEADER LINE' file
sed '$a\FOOTER LINE' file

# Insert blank lines
sed 'G' file                           # After every line
sed '/pattern/G' file                  # After pattern matches
sed '/pattern/{G;G;}' file             # Two blanks after pattern
```

### Configuration File Editing

```bash
# Change configuration value
sed 's/^DEBUG=.*/DEBUG=true/' config.ini
sed 's/^\(PORT=\).*/\1 8080/' config

# Comment out lines
sed 's/^/# /' file                     # Prefix all lines
sed '/pattern/s/^/# /' file            # Comment lines matching pattern
sed '10,20s/^/# /' file                # Comment lines 10-20

# Uncomment lines
sed 's/^# //' file                     # Remove # prefix
sed '/pattern/s/^# *//' file           # Uncomment pattern matches

# Add configuration if not exists
sed '/^MAX_CONN/!s/$/\nMAX_CONN=100/' file
```

### Log File Processing

```bash
# Extract errors
sed -n '/ERROR/p' logfile
sed -n '/ERROR\|FATAL/p' logfile       # ERROR or FATAL

# Filter by timestamp
sed -n '/2024-01-15/p' logfile
sed -n '/2024-01-15 09:/,/2024-01-15 10:/p' logfile

# Remove timestamps
sed 's/^[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} [0-9]\{2\}:[0-9]\{2\}:[0-9]\{2\} //' logfile

# Anonymize IP addresses
sed 's/[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}/XXX.XXX.XXX.XXX/g' logfile

# Count error types
sed -n 's/.*ERROR: \([^:]*\).*/\1/p' logfile | sort | uniq -c
```

### CSV/TSV Processing

```bash
# Change delimiter
sed 's/,/\t/g' file.csv                # CSV to TSV
sed 's/\t/,/g' file.tsv                # TSV to CSV

# Extract specific columns (simple cases)
sed 's/^\([^,]*\),\([^,]*\).*/\1,\2/' file.csv  # First two columns

# Remove quotes
sed 's/"//g' file.csv

# Add quotes to fields
sed 's/\([^,]*\)/"\1"/g' file.csv

# Remove header
sed '1d' file.csv

# Add header
sed '1i\Name,Age,Email' file.csv
```

### URL/Path Manipulation

```bash
# Change protocol
sed 's|http://|https://|g' file

# Extract domain
sed 's|.*://\([^/]*\).*|\1|' urls.txt

# Extract path
sed 's|.*://[^/]*/\(.*\)|\1|' urls.txt

# Change file extension
sed 's/\.txt$/.md/' files.txt
sed 's/\.[^.]*$/.new/' files.txt       # Any extension to .new

# Convert Windows paths to Unix
sed 's|\\|/|g' paths.txt
sed 's|C:|/mnt/c|g' paths.txt
```

## Practical Examples

### Example 1: Cleanup and Format Source Code

```bash
# Remove trailing whitespace
sed 's/[[:space:]]*$//' source.cpp

# Convert tabs to spaces
sed 's/\t/    /g' source.cpp

# Remove C++ comments
sed 's|//.*||' source.cpp

# Format function definitions
sed '/^[a-zA-Z].*{$/i\\' source.cpp    # Add blank before {
```

### Example 2: Process Apache Access Logs

```bash
# Extract IP addresses
sed 's/^\([0-9.]*\).*/\1/' access.log

# Filter by status code
sed -n '/ 404 /p' access.log           # 404 errors
sed -n '/ [45][0-9][0-9] /p' access.log  # All errors

# Extract request paths
sed 's/.*"\(GET\|POST\) \([^ ]*\).*/\2/' access.log

# Anonymize IPs
sed 's/\([0-9]\{1,3\}\.\)[0-9]\{1,3\}\.\([0-9]\{1,3\}\.\)[0-9]\{1,3\}/\1XXX.\2XXX/' access.log
```

### Example 3: HTML Processing

```bash
# Remove HTML tags
sed 's/<[^>]*>//g' page.html

# Extract links
sed -n 's/.*href="\([^"]*\)".*/\1/p' page.html

# Convert to text entities
sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g' file.txt

# Extract title
sed -n 's/.*<title>\(.*\)<\/title>.*/\1/p' page.html
```

### Example 4: Email Processing

```bash
# Extract email addresses
sed -n 's/.*\([a-zA-Z0-9._%+-]*@[a-zA-Z0-9.-]*\.[a-zA-Z]\{2,\}\).*/\1/p' file.txt

# Obfuscate emails
sed 's/@/ [at] /g; s/\./ [dot] /g' emails.txt

# Validate email format (simple)
sed -n '/^[a-zA-Z0-9._%+-]\+@[a-zA-Z0-9.-]\+\.[a-zA-Z]\{2,\}$/p' emails.txt
```

### Example 5: Data Transformation

```bash
# Uppercase/lowercase
sed 's/.*/\U&/' file                   # Convert to uppercase
sed 's/.*/\L&/' file                   # Convert to lowercase
sed 's/\b\(.\)/\U\1/g' file            # Capitalize words

# Format phone numbers
sed 's/\([0-9]\{3\}\)\([0-9]\{3\}\)\([0-9]\{4\}\)/(\1) \2-\3/' phones.txt

# Format dates
sed 's|\([0-9]\{2\}\)/\([0-9]\{2\}\)/\([0-9]\{4\}\)|\3-\1-\2|' dates.txt

# Add thousand separators
sed ':a; s/\([0-9]\)\([0-9]\{3\}\)\($\|,\)/\1,\2\3/; ta' numbers.txt
```

### Example 6: Script Generation

```bash
# Generate SQL INSERT statements
sed 's/\(.*\),\(.*\),\(.*\)/INSERT INTO users VALUES ("\1", "\2", "\3");/' data.csv

# Generate test cases
sed 's/\(.*\)/test("\1", function() { ... });/' testcases.txt

# Generate markdown list
sed 's/^/- /' items.txt
sed 's/^\(.*\)$/- [\1](\1.md)/' files.txt
```

## sed Scripts

For complex operations, use sed script files:

```bash
# script.sed
/^#/d                    # Remove comments
/^$/d                    # Remove empty lines
s/  */ /g                # Collapse spaces
s/^ //                   # Remove leading space
s/ $//                   # Remove trailing space
```

```bash
# Run script
sed -f script.sed input.txt

# Combine script with commands
sed -f script.sed -e 's/extra/replacement/' input.txt
```

### Complex Script Example

```bash
# format-code.sed - Format C code
# Remove trailing whitespace
s/[[:space:]]*$//

# Convert tabs to 4 spaces
s/\t/    /g

# Add space after keywords
s/\(if\|for\|while\)(/\1 (/g

# Format braces
/^[[:space:]]*{/i\

# Remove multiple blank lines
/^$/{
    N
    /^\n$/D
}
```

## In-Place Editing

### Basic In-Place Editing

```bash
# Edit file directly
sed -i 's/old/new/g' file.txt

# Create backup with extension
sed -i.bak 's/old/new/g' file.txt      # Creates file.txt.bak
sed -i.backup 's/old/new/g' file.txt   # Creates file.txt.backup

# Edit multiple files
sed -i 's/old/new/g' *.txt

# Edit with backup (BSD/macOS)
sed -i '' 's/old/new/g' file.txt       # No backup
sed -i '.bak' 's/old/new/g' file.txt   # With backup
```

### Safe In-Place Editing

```bash
# Test before applying
sed 's/old/new/g' file.txt | diff file.txt -

# Conditional in-place edit
if sed 's/old/new/g' file.txt > temp.txt; then
    mv temp.txt file.txt
else
    rm temp.txt
    echo "sed failed"
fi

# Atomic replacement
sed 's/old/new/g' file.txt > file.txt.tmp && mv file.txt.tmp file.txt
```

## Best Practices

### 1. Quote Your Patterns

```bash
# Good
sed 's/pattern/replacement/' file

# Bad (can cause issues with special characters)
sed s/pattern/replacement/ file
```

### 2. Use Appropriate Delimiters

```bash
# Hard to read
sed 's/\/usr\/local\/bin/\/opt\/bin/g' file

# Better
sed 's|/usr/local/bin|/opt/bin|g' file
```

### 3. Test Before In-Place Edit

```bash
# Test output first
sed 's/old/new/g' file.txt

# Then apply in-place
sed -i 's/old/new/g' file.txt
```

### 4. Use -n with p for Filtering

```bash
# Wrong (prints lines twice)
sed '/pattern/p' file

# Correct
sed -n '/pattern/p' file
```

### 5. Escape Special Characters

```bash
# Literal dot
sed 's/\./DOT/g' file

# Literal asterisk
sed 's/\*/STAR/g' file

# Variables (use double quotes)
VAR="value"
sed "s/pattern/$VAR/" file
```

### 6. Use Extended Regex When Needed

```bash
# Basic regex (need to escape +, ?, |, etc.)
sed 's/a\+/x/' file

# Extended regex (GNU sed)
sed -r 's/a+/x/' file
sed -E 's/a+/x/' file        # POSIX/BSD compatible
```

### 7. Process Large Files Efficiently

```bash
# Stop after first match
sed '/pattern/q' large-file.txt

# Process specific range only
sed -n '1000,2000p' large-file.txt

# Use quit to limit processing
sed '1000q' large-file.txt
```

## Common Gotchas

### 1. Greedy Matching

```bash
# Problem: Greedy matching
echo "abc123def456" | sed 's/.*[0-9]/X/'    # Returns: X6

# Solution: Non-greedy (use multiple steps)
echo "abc123def456" | sed 's/[0-9]\+/X/g'   # Returns: abcXdefX
```

### 2. Line Endings

```bash
# DOS to Unix
sed 's/\r$//' dosfile.txt

# Unix to DOS
sed 's/$/\r/' unixfile.txt
```

### 3. Backreferences

```bash
# GNU sed uses \1, \2, etc.
sed 's/\(.*\):\(.*\)/\2:\1/' file

# Some versions use &, \&
```

### 4. In-Place Editing Differences

```bash
# GNU sed
sed -i 's/old/new/' file           # No backup
sed -i.bak 's/old/new/' file       # With backup

# BSD/macOS sed
sed -i '' 's/old/new/' file        # No backup
sed -i '.bak' 's/old/new/' file    # With backup
```

### 5. Empty Pattern Space

```bash
# Problem: Can't delete all lines and add new content
sed 'd; a\new text' file           # Won't work

# Solution: Use c to change
sed 'c\new text' file
```

## sed Versions and Compatibility

### GNU sed vs BSD sed

```bash
# Extended regex
sed -r 's/pattern/replacement/' file    # GNU
sed -E 's/pattern/replacement/' file    # BSD/POSIX

# In-place editing
sed -i 's/pattern/replacement/' file    # GNU
sed -i '' 's/pattern/replacement/' file # BSD

# Address ranges
sed '1~2d' file                         # GNU (every other line)
# BSD requires different approach
```

### Portable sed Scripts

```bash
# Use POSIX features only
# Avoid GNU-specific features:
# - Extended regex (-r)
# - In-place editing without extension
# - Address stepping (1~2)
# - \U, \L for case conversion

# Test on multiple platforms
# Use sed -i.bak for portability
# Avoid relying on GNU-specific escapes
```

## Performance Tips

### 1. Early Exit

```bash
# Stop processing after finding first match
sed '/pattern/q' large-file.txt

# Process only needed range
sed -n '100,200p' large-file.txt
```

### 2. Combine Multiple Edits

```bash
# Less efficient
sed 's/foo/bar/' file | sed 's/baz/qux/'

# More efficient
sed 's/foo/bar/; s/baz/qux/' file
```

### 3. Use Specific Addresses

```bash
# Less efficient (checks all lines)
sed 's/pattern/replacement/g' file

# More efficient (only specific range)
sed '10,1000s/pattern/replacement/g' file
```

## Quick Reference

### Commands

| Command | Description |
|---------|-------------|
| `s/pattern/replacement/` | Substitute |
| `d` | Delete |
| `p` | Print |
| `n` | Next line |
| `N` | Append next line |
| `a\text` | Append after |
| `i\text` | Insert before |
| `c\text` | Change (replace) |
| `q` | Quit |
| `r file` | Read file |
| `w file` | Write to file |
| `y/src/dst/` | Transform |
| `h` | Copy to hold space |
| `H` | Append to hold space |
| `g` | Get from hold space |
| `G` | Append from hold space |
| `x` | Exchange spaces |
| `=` | Print line number |

### Flags

| Flag | Description |
|------|-------------|
| `g` | Global (all occurrences) |
| `i` | Case-insensitive |
| `p` | Print |
| `w file` | Write to file |
| `1,2,3...` | Nth occurrence |

### Options

| Option | Description |
|--------|-------------|
| `-n` | Suppress automatic output |
| `-i[ext]` | In-place editing |
| `-e cmd` | Add command |
| `-f file` | Read commands from file |
| `-r` | Extended regex (GNU) |
| `-E` | Extended regex (POSIX/BSD) |
| `--debug` | Print program with annotations |

### Regular Expression Syntax

| Pattern | Description |
|---------|-------------|
| `.` | Any character |
| `^` | Start of line |
| `$` | End of line |
| `*` | Zero or more |
| `\+` | One or more |
| `\?` | Zero or one |
| `\{n\}` | Exactly n |
| `\{n,\}` | n or more |
| `\{n,m\}` | Between n and m |
| `[...]` | Character class |
| `[^...]` | Negated class |
| `\(` ... `\)` | Grouping |
| `\1`, `\2` | Backreferences |
| `\|` | Alternation |
| `\<`, `\>` | Word boundaries |
| `\b` | Word boundary (GNU) |

### Common Patterns

```bash
# Replace
sed 's/old/new/g' file

# Delete
sed '/pattern/d' file

# Print matching
sed -n '/pattern/p' file

# Insert
sed '1i\text' file

# Append
sed '$a\text' file

# Extract lines
sed -n '10,20p' file

# In-place edit
sed -i 's/old/new/g' file
```

## Troubleshooting

### Debug Your sed Commands

```bash
# Print what sed is doing (GNU sed)
sed --debug 's/pattern/replacement/' file

# Trace execution
sed -n 'l' file                    # Show special characters

# Test patterns separately
sed -n '/pattern/=' file           # Print line numbers of matches
```

### Common Error Messages

```bash
# "unterminated `s' command"
# Forgot closing delimiter
sed 's/pattern/replacement' file   # Wrong
sed 's/pattern/replacement/' file  # Correct

# "invalid reference \1 on `s' command's RHS"
# No capturing group in pattern
sed 's/pattern/\1/' file           # Wrong
sed 's/\(pattern\)/\1/' file       # Correct

# "extra characters after command"
# Missing semicolon between commands
sed 's/a/b/ s/c/d/' file           # Wrong
sed 's/a/b/; s/c/d/' file          # Correct
```

### Testing and Validation

```bash
# Dry run - see changes before applying
sed 's/old/new/g' file > /dev/null && echo "Syntax OK"

# Compare before/after
diff <(cat original.txt) <(sed 's/old/new/' original.txt)

# Count changes
sed -n 's/old/new/p' file | wc -l

# Validate regex pattern
echo "test" | sed '/pattern/!d'
```

sed is an essential tool for text processing and manipulation. Master these patterns and techniques to efficiently handle automated text transformations, configuration management, and data processing tasks.
