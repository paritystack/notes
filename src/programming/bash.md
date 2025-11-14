# Bash Programming

## Overview

Bash (Bourne Again SHell) is a Unix shell and command language used for automating tasks, system administration, and scripting. It's the default shell on most Linux distributions and macOS.

**Key Features:**
- Command execution and scripting
- Text processing and file manipulation
- Process control and job management
- Environment variable management
- Piping and redirection
- Pattern matching and globbing

---

## Basic Syntax

### Variables

```bash
# Variable assignment (no spaces around =)
name="Alice"
age=30
readonly PI=3.14159  # Read-only variable

# Accessing variables
echo "Hello, $name"
echo "Hello, ${name}!"  # Recommended for clarity

# Command substitution
current_date=$(date)
current_dir=`pwd`  # Old style, avoid

# Default values
echo "${var:-default}"      # Use default if var is unset
echo "${var:=default}"      # Set var to default if unset
echo "${var:+alternate}"    # Use alternate if var is set
echo "${var:?error message}"  # Error if var is unset

# String length
name="Alice"
echo "${#name}"  # 5

# Substring
echo "${name:0:3}"  # Ali
```

### Data Types

```bash
# Strings
str="Hello World"
str='Single quotes - literal'
str="Double quotes - $variable expansion"

# Arrays
fruits=("apple" "banana" "cherry")
echo "${fruits[0]}"      # apple
echo "${fruits[@]}"      # All elements
echo "${#fruits[@]}"     # Array length
fruits+=("date")         # Append

# Associative arrays (Bash 4+)
declare -A person
person[name]="Alice"
person[age]=30
echo "${person[name]}"

# Integers
declare -i num=42
num=$num+10  # Arithmetic
```

---

## Control Flow

### If Statements

```bash
# Basic if
if [ "$age" -gt 18 ]; then
    echo "Adult"
fi

# If-elif-else
if [ "$age" -lt 13 ]; then
    echo "Child"
elif [ "$age" -lt 20 ]; then
    echo "Teenager"
else
    echo "Adult"
fi

# String comparison
if [ "$name" = "Alice" ]; then
    echo "Hello Alice"
fi

if [ "$name" != "Bob" ]; then
    echo "Not Bob"
fi

# File tests
if [ -f "file.txt" ]; then
    echo "File exists"
fi

if [ -d "directory" ]; then
    echo "Directory exists"
fi

if [ -r "file.txt" ]; then
    echo "File is readable"
fi

# Logical operators
if [ "$age" -gt 18 ] && [ "$age" -lt 65 ]; then
    echo "Working age"
fi

if [ "$age" -lt 18 ] || [ "$age" -gt 65 ]; then
    echo "Not working age"
fi

# Modern test syntax [[ ]]
if [[ "$name" == "Alice" ]]; then
    echo "Hello Alice"
fi

if [[ "$name" =~ ^A ]]; then  # Regex matching
    echo "Name starts with A"
fi
```

### Comparison Operators

```bash
# Numeric comparison
[ "$a" -eq "$b" ]  # Equal
[ "$a" -ne "$b" ]  # Not equal
[ "$a" -gt "$b" ]  # Greater than
[ "$a" -ge "$b" ]  # Greater than or equal
[ "$a" -lt "$b" ]  # Less than
[ "$a" -le "$b" ]  # Less than or equal

# String comparison
[ "$a" = "$b" ]    # Equal
[ "$a" != "$b" ]   # Not equal
[ -z "$a" ]        # String is empty
[ -n "$a" ]        # String is not empty

# File tests
[ -e file ]        # Exists
[ -f file ]        # Regular file
[ -d file ]        # Directory
[ -r file ]        # Readable
[ -w file ]        # Writable
[ -x file ]        # Executable
[ -s file ]        # Not empty
[ file1 -nt file2 ]  # file1 newer than file2
[ file1 -ot file2 ]  # file1 older than file2
```

### Loops

```bash
# For loop
for i in 1 2 3 4 5; do
    echo "$i"
done

# C-style for loop
for ((i=0; i<5; i++)); do
    echo "$i"
done

# For loop with range
for i in {1..10}; do
    echo "$i"
done

# For loop with step
for i in {0..10..2}; do
    echo "$i"  # 0, 2, 4, 6, 8, 10
done

# Iterate over array
fruits=("apple" "banana" "cherry")
for fruit in "${fruits[@]}"; do
    echo "$fruit"
done

# Iterate over files
for file in *.txt; do
    echo "Processing $file"
done

# While loop
count=0
while [ $count -lt 5 ]; do
    echo "$count"
    ((count++))
done

# Read file line by line
while IFS= read -r line; do
    echo "$line"
done < file.txt

# Until loop
count=0
until [ $count -ge 5 ]; do
    echo "$count"
    ((count++))
done

# Break and continue
for i in {1..10}; do
    if [ $i -eq 5 ]; then
        continue  # Skip 5
    fi
    if [ $i -eq 8 ]; then
        break     # Stop at 8
    fi
    echo "$i"
done
```

### Case Statements

```bash
case "$1" in
    start)
        echo "Starting service..."
        ;;
    stop)
        echo "Stopping service..."
        ;;
    restart)
        echo "Restarting service..."
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac

# Pattern matching in case
case "$filename" in
    *.txt)
        echo "Text file"
        ;;
    *.jpg|*.png)
        echo "Image file"
        ;;
    *)
        echo "Unknown file type"
        ;;
esac
```

---

## Functions

```bash
# Basic function
greet() {
    echo "Hello, $1!"
}

greet "Alice"  # Hello, Alice!

# Function with return value
add() {
    local result=$(($1 + $2))
    echo "$result"
}

sum=$(add 5 3)
echo "Sum: $sum"

# Function with return code
check_file() {
    if [ -f "$1" ]; then
        return 0  # Success
    else
        return 1  # Failure
    fi
}

if check_file "file.txt"; then
    echo "File exists"
else
    echo "File not found"
fi

# Local variables
my_function() {
    local local_var="I'm local"
    global_var="I'm global"
}

# Function with multiple return values
get_stats() {
    local min=1
    local max=100
    local avg=50
    echo "$min $max $avg"
}

read min max avg <<< $(get_stats)
echo "Min: $min, Max: $max, Avg: $avg"
```

---

## String Manipulation

```bash
# Length
str="Hello World"
echo "${#str}"  # 11

# Substring
echo "${str:0:5}"   # Hello
echo "${str:6}"     # World
echo "${str: -5}"   # World (note space before -)

# Replace
echo "${str/World/Universe}"  # Hello Universe (first occurrence)
echo "${str//o/O}"            # HellO WOrld (all occurrences)

# Remove prefix/suffix
filename="example.tar.gz"
echo "${filename#*.}"      # tar.gz (remove shortest prefix)
echo "${filename##*.}"     # gz (remove longest prefix)
echo "${filename%.*}"      # example.tar (remove shortest suffix)
echo "${filename%%.*}"     # example (remove longest suffix)

# Upper/Lower case
str="Hello World"
echo "${str^^}"  # HELLO WORLD
echo "${str,,}"  # hello world
echo "${str^}"   # Hello world (first char upper)

# Trim whitespace
str="  hello  "
str="${str#"${str%%[![:space:]]*}"}"  # Trim left
str="${str%"${str##*[![:space:]]}"}"  # Trim right
```

---

## Input/Output

### Reading Input

```bash
# Read from user
read -p "Enter your name: " name
echo "Hello, $name!"

# Read with timeout
if read -t 5 -p "Enter value (5s timeout): " value; then
    echo "You entered: $value"
else
    echo "Timeout!"
fi

# Read password (hidden)
read -s -p "Enter password: " password
echo

# Read multiple values
read -p "Enter name and age: " name age

# Read into array
IFS=',' read -ra array <<< "apple,banana,cherry"
```

### Output

```bash
# Echo
echo "Hello World"
echo -n "No newline"
echo -e "Line1\nLine2"  # Enable escape sequences

# Printf (more control)
printf "Name: %s, Age: %d\n" "Alice" 30
printf "%.2f\n" 3.14159  # 3.14

# Here document
cat << EOF
This is a
multi-line
message
EOF

# Here string
grep "pattern" <<< "string to search"
```

### Redirection

```bash
# Output redirection
echo "Hello" > file.txt          # Overwrite
echo "World" >> file.txt         # Append

# Input redirection
while read line; do
    echo "$line"
done < file.txt

# Error redirection
command 2> error.log             # Redirect stderr
command > output.txt 2>&1        # Redirect both stdout and stderr
command &> all_output.txt        # Same as above (Bash 4+)

# Discard output
command > /dev/null 2>&1

# Pipe
cat file.txt | grep "pattern" | sort | uniq

# Tee (write to file and stdout)
echo "Hello" | tee file.txt

# Process substitution
diff <(ls dir1) <(ls dir2)
```

---

## File Operations

```bash
# Create file
touch file.txt
echo "content" > file.txt

# Copy
cp source.txt dest.txt
cp -r source_dir/ dest_dir/

# Move/Rename
mv old.txt new.txt
mv file.txt directory/

# Delete
rm file.txt
rm -r directory/
rm -f file.txt  # Force delete

# Create directory
mkdir directory
mkdir -p path/to/nested/directory

# Read file
cat file.txt
head -n 10 file.txt  # First 10 lines
tail -n 10 file.txt  # Last 10 lines
tail -f file.txt     # Follow file (live updates)

# File permissions
chmod 755 script.sh      # rwxr-xr-x
chmod +x script.sh       # Add execute permission
chmod u+x script.sh      # User execute
chmod go-w file.txt      # Remove write for group and others

# File ownership
chown user:group file.txt

# Find files
find . -name "*.txt"
find . -type f -name "*.log"
find . -mtime -7  # Modified in last 7 days
find . -size +10M  # Larger than 10MB
```

---

## Process Management

```bash
# Run in background
command &

# List jobs
jobs

# Bring to foreground
fg %1

# Send to background
bg %1

# Kill process
kill PID
kill -9 PID  # Force kill
killall process_name

# Process info
ps aux
ps aux | grep process_name
top
htop

# Exit status
command
echo $?  # 0 = success, non-zero = failure

# Conditional execution
command1 && command2  # command2 runs if command1 succeeds
command1 || command2  # command2 runs if command1 fails
command1 ; command2   # command2 runs regardless

# Wait for process
command &
PID=$!
wait $PID
```

---

## Common Patterns

### Error Handling

```bash
# Exit on error
set -e  # Exit if any command fails
set -u  # Exit if undefined variable is used
set -o pipefail  # Exit if any command in pipe fails

# Combined
set -euo pipefail

# Error function
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

[ -f "file.txt" ] || error_exit "File not found"

# Trap errors
trap 'echo "Error on line $LINENO"' ERR

# Cleanup on exit
cleanup() {
    rm -f /tmp/tempfile
}
trap cleanup EXIT
```

### Argument Parsing

```bash
# Positional arguments
echo "Script: $0"
echo "First arg: $1"
echo "Second arg: $2"
echo "All args: $@"
echo "Number of args: $#"

# Shift arguments
while [ $# -gt 0 ]; do
    echo "$1"
    shift
done

# Parse options
while getopts "a:b:c" opt; do
    case $opt in
        a)
            echo "Option -a: $OPTARG"
            ;;
        b)
            echo "Option -b: $OPTARG"
            ;;
        c)
            echo "Option -c"
            ;;
        \?)
            echo "Invalid option: -$OPTARG"
            exit 1
            ;;
    esac
done

# Long options
while [ $# -gt 0 ]; do
    case "$1" in
        --help)
            echo "Usage: $0 [options]"
            exit 0
            ;;
        --file=*)
            FILE="${1#*=}"
            ;;
        --verbose)
            VERBOSE=1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done
```

### Logging

```bash
# Simple logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Script started"

# Log levels
LOG_LEVEL=${LOG_LEVEL:-INFO}

log_debug() {
    [ "$LOG_LEVEL" = "DEBUG" ] && echo "[DEBUG] $1"
}

log_info() {
    echo "[INFO] $1"
}

log_error() {
    echo "[ERROR] $1" >&2
}

# Log to file
exec > >(tee -a script.log)
exec 2>&1
```

### Configuration Files

```bash
# Source configuration
CONFIG_FILE="config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# config.sh
# DB_HOST="localhost"
# DB_PORT=5432
# DB_NAME="mydb"

# Read key-value pairs
while IFS='=' read -r key value; do
    case "$key" in
        DB_HOST) DB_HOST="$value" ;;
        DB_PORT) DB_PORT="$value" ;;
        DB_NAME) DB_NAME="$value" ;;
    esac
done < config.txt
```

---

## Text Processing

```bash
# grep (search)
grep "pattern" file.txt
grep -i "pattern" file.txt        # Case insensitive
grep -r "pattern" directory/      # Recursive
grep -v "pattern" file.txt        # Invert match
grep -n "pattern" file.txt        # Show line numbers
grep -c "pattern" file.txt        # Count matches
grep -E "regex" file.txt          # Extended regex

# sed (stream editor)
sed 's/old/new/' file.txt         # Replace first occurrence
sed 's/old/new/g' file.txt        # Replace all
sed -i 's/old/new/g' file.txt     # In-place edit
sed -n '10,20p' file.txt          # Print lines 10-20
sed '/pattern/d' file.txt         # Delete matching lines

# awk (text processing)
awk '{print $1}' file.txt         # Print first column
awk '{print $1, $3}' file.txt     # Print columns 1 and 3
awk -F: '{print $1}' /etc/passwd  # Custom delimiter
awk '$3 > 100' file.txt           # Filter rows
awk '{sum += $1} END {print sum}' file.txt  # Sum column

# cut (extract columns)
cut -d: -f1 /etc/passwd           # Field 1, delimiter :
cut -c1-10 file.txt               # Characters 1-10

# sort
sort file.txt
sort -r file.txt                  # Reverse
sort -n file.txt                  # Numeric sort
sort -k2 file.txt                 # Sort by column 2
sort -u file.txt                  # Unique

# uniq (unique lines)
sort file.txt | uniq              # Remove duplicates
sort file.txt | uniq -c           # Count occurrences
sort file.txt | uniq -d           # Only duplicates

# wc (word count)
wc -l file.txt                    # Line count
wc -w file.txt                    # Word count
wc -c file.txt                    # Byte count

# tr (translate characters)
echo "hello" | tr 'a-z' 'A-Z'     # HELLO
echo "hello123" | tr -d '0-9'     # hello

# fold
echo "hell" | fold -w 2	  	  # he and ll
```

---

## Common Utilities

```bash
# Date and time
date                              # Current date/time
date '+%Y-%m-%d'                  # 2024-01-15
date '+%Y-%m-%d %H:%M:%S'         # 2024-01-15 14:30:00
date -d "yesterday"               # Yesterday's date
date -d "+7 days"                 # Date 7 days from now

# Arithmetic
echo $((5 + 3))                   # 8
echo $((10 / 3))                  # 3 (integer division)
echo "scale=2; 10 / 3" | bc       # 3.33 (bc for floating point)

# Random numbers
echo $RANDOM                      # Random number 0-32767
echo $((RANDOM % 100))            # Random 0-99

# Sleep
sleep 5                           # Sleep 5 seconds
sleep 0.5                         # Sleep 0.5 seconds

# Command existence check
if command -v git &> /dev/null; then
    echo "Git is installed"
fi

# Array operations
arr=(1 2 3 4 5)
echo "${arr[@]}"                  # All elements
echo "${#arr[@]}"                 # Length
echo "${arr[@]:1:3}"              # Slice [1:4]
arr+=(6)                          # Append
```

---

## Best Practices

1. **Always quote variables**: `"$var"` not `$var`
2. **Use `set -euo pipefail`** for safer scripts
3. **Check command existence** before using
4. **Validate input** and arguments
5. **Use functions** for code reuse
6. **Add comments** and documentation
7. **Use meaningful variable names**
8. **Handle errors** explicitly
9. **Use `[[` instead of `[`** for conditions
10. **Avoid parsing `ls` output** - use globbing or `find`

---

## Script Template

```bash
#!/usr/bin/env bash

# Script: script_name.sh
# Description: What this script does
# Author: Your Name
# Date: 2024-01-15

set -euo pipefail  # Exit on error, undefined var, pipe failure

# Constants
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"

# Variables
VERBOSE=0
DRY_RUN=0

# Functions
usage() {
    cat << EOF
Usage: $SCRIPT_NAME [OPTIONS]

Description of what the script does.

OPTIONS:
    -h, --help      Show this help message
    -v, --verbose   Verbose output
    -n, --dry-run   Dry run mode
EOF
}

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

cleanup() {
    log_info "Cleaning up..."
    # Cleanup code here
}

main() {
    log_info "Starting script..."

    # Main script logic here

    log_info "Script completed successfully"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Trap cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"
```

---

## Common Use Cases

### Backup Script

```bash
#!/bin/bash

BACKUP_DIR="/backup"
SOURCE_DIR="/data"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${DATE}.tar.gz"

tar -czf "${BACKUP_DIR}/${BACKUP_FILE}" "${SOURCE_DIR}"
echo "Backup created: ${BACKUP_FILE}"

# Keep only last 7 backups
cd "${BACKUP_DIR}"
ls -t backup_*.tar.gz | tail -n +8 | xargs -r rm
```

### System Monitoring

```bash
#!/bin/bash

CPU_THRESHOLD=80
DISK_THRESHOLD=90

# Check CPU usage
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
if (( $(echo "$CPU_USAGE > $CPU_THRESHOLD" | bc -l) )); then
    echo "WARNING: CPU usage is ${CPU_USAGE}%"
fi

# Check disk usage
df -H | grep -vE '^Filesystem|tmpfs|cdrom' | awk '{print $5 " " $1}' | while read output; do
    usage=$(echo $output | awk '{print $1}' | sed 's/%//g')
    partition=$(echo $output | awk '{print $2}')
    if [ $usage -ge $DISK_THRESHOLD ]; then
        echo "WARNING: Disk usage on $partition is ${usage}%"
    fi
done
```

---

## Useful Resources

- **ShellCheck**: Linter for shell scripts
- **Bash Manual**: `man bash`
- **Bash Guide**: https://mywiki.wooledge.org/BashGuide
- **Explainshell**: Explain shell commands (explainshell.com)
