# AWK

AWK is a powerful pattern-scanning and text-processing language named after its creators: Aho, Weinberger, and Kernighan. It excels at processing structured text files, extracting data, and generating reports.

## Overview

AWK reads input line by line, splits each line into fields, and allows you to process data using pattern-action statements. It's particularly useful for log analysis, data extraction, and text transformation.

**Key Features:**
- Pattern-action programming model
- Built-in field splitting
- Associative arrays
- Regular expression support
- Built-in variables and functions
- C-like syntax
- No compilation required

**Common Use Cases:**
- Log file analysis
- CSV/TSV processing
- Data extraction and reformatting
- Report generation
- Configuration file processing
- System administration tasks

---

## Basic Syntax

### Program Structure

```bash
awk 'pattern { action }' file

# Pattern: Condition to match
# Action: Commands to execute when pattern matches
# If pattern is omitted, action applies to all lines
# If action is omitted, matching lines are printed
```

### Simple Examples

```bash
# Print all lines (like cat)
awk '{ print }' file.txt

# Print all lines (default action)
awk '1' file.txt

# Print specific field
awk '{ print $1 }' file.txt

# Print multiple fields
awk '{ print $1, $3 }' file.txt

# Print entire line
awk '{ print $0 }' file.txt
```

---

## Fields and Records

### Field Basics

AWK automatically splits each line into fields based on whitespace.

```bash
# Example input: "Alice 25 Engineer"

# Print first field
awk '{ print $1 }' file        # Alice

# Print second field
awk '{ print $2 }' file        # 25

# Print last field
awk '{ print $NF }' file       # Engineer

# Print second-to-last field
awk '{ print $(NF-1) }' file   # 25

# Print all fields
awk '{ print $0 }' file        # Alice 25 Engineer

# Number of fields
awk '{ print NF }' file        # 3
```

### Field Separators

```bash
# Default separator (whitespace)
awk '{ print $1 }' file

# Custom separator (colon)
awk -F: '{ print $1 }' /etc/passwd

# Multiple character separator
awk -F'::' '{ print $1 }' file

# Regex separator
awk -F'[,:]' '{ print $1 }' file

# Tab separator
awk -F'\t' '{ print $1 }' file

# Set separator in BEGIN
awk 'BEGIN { FS=":" } { print $1 }' file

# Output field separator
awk 'BEGIN { OFS="," } { print $1, $2 }' file
```

---

## Built-in Variables

### Automatic Variables

```bash
# NR - Number of Records (line number)
awk '{ print NR, $0 }' file

# NF - Number of Fields
awk '{ print NF, $0 }' file

# FNR - File Number of Records (resets for each file)
awk '{ print FNR, $0 }' file1 file2

# FS - Field Separator (input)
awk 'BEGIN { FS=":" } { print $1 }' file

# OFS - Output Field Separator
awk 'BEGIN { OFS="|" } { print $1, $2 }' file

# RS - Record Separator (input, default: newline)
awk 'BEGIN { RS=";" } { print }' file

# ORS - Output Record Separator (default: newline)
awk 'BEGIN { ORS="; " } { print }' file

# FILENAME - Current filename
awk '{ print FILENAME, $0 }' file

# ARGC - Argument count
awk 'BEGIN { print ARGC }' file1 file2

# ARGV - Argument array
awk 'BEGIN { for(i=0; i<ARGC; i++) print ARGV[i] }' file
```

### Example: Line Numbers

```bash
# Print with line numbers
awk '{ print NR ":", $0 }' file

# Print only specific lines
awk 'NR==5' file                # Line 5
awk 'NR>=10 && NR<=20' file     # Lines 10-20
awk 'NR==1 || NR==10' file      # Lines 1 and 10
```

---

## Pattern Matching

### Basic Patterns

```bash
# Match lines containing "error"
awk '/error/' file

# Case-insensitive match
awk 'tolower($0) ~ /error/' file

# Match lines NOT containing "error"
awk '!/error/' file

# Match specific field
awk '$3 ~ /error/' file         # Field 3 contains "error"
awk '$3 !~ /error/' file        # Field 3 doesn't contain "error"

# Match exact field
awk '$1 == "ERROR"' file
awk '$1 != "ERROR"' file
```

### Regular Expressions

```bash
# Lines starting with "Error"
awk '/^Error/' file

# Lines ending with "failed"
awk '/failed$/' file

# Lines with numbers
awk '/[0-9]+/' file

# Email addresses
awk '/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/' file

# IP addresses
awk '/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/' file

# Match alternatives
awk '/error|warning|critical/' file

# Match with groups
awk '/^(GET|POST|PUT|DELETE)/' access.log
```

### Comparison Operators

```bash
# Numeric comparisons
awk '$2 > 100' file             # Field 2 greater than 100
awk '$2 >= 100' file            # Greater or equal
awk '$2 < 100' file             # Less than
awk '$2 <= 100' file            # Less or equal
awk '$2 == 100' file            # Equal to
awk '$2 != 100' file            # Not equal

# String comparisons
awk '$1 == "ERROR"' file
awk '$1 != "ERROR"' file

# Multiple conditions (AND)
awk '$2 > 100 && $3 < 50' file

# Multiple conditions (OR)
awk '$1 == "ERROR" || $1 == "WARN"' file

# NOT operator
awk '!($2 > 100)' file
```

### Range Patterns

```bash
# Between two patterns
awk '/START/,/END/' file

# From pattern to end
awk '/START/,0' file

# From line 10 to 20
awk 'NR==10,NR==20' file

# From first match to second match
awk '/begin/,/end/ { print NR, $0 }' file
```

---

## Actions and Statements

### Print Statements

```bash
# Basic print
awk '{ print $1 }' file

# Print with text
awk '{ print "Name:", $1 }' file

# Print multiple fields
awk '{ print $1, $2, $3 }' file

# Print with custom separator
awk '{ print $1 "|" $2 "|" $3 }' file

# Formatted print (printf)
awk '{ printf "%-10s %5d\n", $1, $2 }' file

# Print to file
awk '{ print $0 > "output.txt" }' file

# Append to file
awk '{ print $0 >> "output.txt" }' file
```

### Printf Formatting

```bash
# String formatting
awk '{ printf "%s\n", $1 }' file
awk '{ printf "%-20s\n", $1 }' file    # Left-aligned, width 20
awk '{ printf "%20s\n", $1 }' file     # Right-aligned, width 20

# Integer formatting
awk '{ printf "%d\n", $2 }' file
awk '{ printf "%5d\n", $2 }' file      # Width 5
awk '{ printf "%05d\n", $2 }' file     # Zero-padded

# Float formatting
awk '{ printf "%.2f\n", $3 }' file     # 2 decimal places
awk '{ printf "%8.2f\n", $3 }' file    # Width 8, 2 decimals

# Hexadecimal
awk '{ printf "%x\n", $1 }' file

# Multiple formats
awk '{ printf "Name: %-10s Age: %3d Salary: %8.2f\n", $1, $2, $3 }' file
```

---

## Variables and Operators

### User-Defined Variables

```bash
# Simple variable
awk '{ count = count + 1 } END { print count }' file

# Multiple variables
awk '{ sum += $1; count++ } END { print sum/count }' file

# String variables
awk '{ name = $1; print "Hello", name }' file

# Initialize in BEGIN
awk 'BEGIN { total = 0 } { total += $1 } END { print total }' file
```

### Arithmetic Operators

```bash
# Addition
awk '{ print $1 + $2 }' file

# Subtraction
awk '{ print $1 - $2 }' file

# Multiplication
awk '{ print $1 * $2 }' file

# Division
awk '{ print $1 / $2 }' file

# Modulo
awk '{ print $1 % $2 }' file

# Exponentiation
awk '{ print $1 ** $2 }' file

# Increment/Decrement
awk '{ count++; print count }' file
awk '{ ++count; print count }' file
awk '{ count--; print count }' file

# Compound assignment
awk '{ sum += $1; count += 1 }' file
```

### String Operators

```bash
# Concatenation
awk '{ print $1 $2 }' file
awk '{ print $1 " " $2 }' file
awk '{ name = $1 " " $2; print name }' file

# Length
awk '{ print length($1) }' file
awk '{ print length }' file            # Length of $0

# Substring
awk '{ print substr($1, 1, 3) }' file  # First 3 chars
awk '{ print substr($1, 4) }' file     # From 4th char

# Index (find position)
awk '{ print index($0, "error") }' file

# Split
awk '{ split($0, arr, ":"); print arr[1] }' file
```

---

## Control Flow

### If-Else Statements

```bash
# Simple if
awk '{ if ($1 > 100) print $0 }' file

# If-else
awk '{ if ($1 > 100) print "High"; else print "Low" }' file

# If-else if-else
awk '{
    if ($1 >= 90) print "A"
    else if ($1 >= 80) print "B"
    else if ($1 >= 70) print "C"
    else print "F"
}' file

# Nested if
awk '{
    if ($1 > 0) {
        if ($1 > 100)
            print "Very High"
        else
            print "Normal"
    }
}' file

# Ternary operator
awk '{ print ($1 > 100) ? "High" : "Low" }' file
```

### Loops

```bash
# For loop
awk '{
    for (i = 1; i <= NF; i++)
        print $i
}' file

# For loop with custom range
awk 'BEGIN {
    for (i = 1; i <= 10; i++)
        print i
}'

# While loop
awk '{
    i = 1
    while (i <= NF) {
        print $i
        i++
    }
}' file

# Do-while loop
awk '{
    i = 1
    do {
        print $i
        i++
    } while (i <= NF)
}' file

# Break and continue
awk '{
    for (i = 1; i <= NF; i++) {
        if ($i == "skip") continue
        if ($i == "stop") break
        print $i
    }
}' file
```

---

## Arrays

### Associative Arrays

```bash
# Simple array
awk '{ count[$1]++ } END { for (word in count) print word, count[word] }' file

# Multi-dimensional array (simulated)
awk '{ arr[$1, $2] = $3 } END { for (key in arr) print key, arr[key] }' file

# Check if element exists
awk '{ if ($1 in count) count[$1]++; else count[$1] = 1 }' file

# Delete array element
awk '{ arr[$1] = $2 } END { delete arr["key"]; for (k in arr) print k, arr[k] }' file

# Array of arrays (split)
awk '{
    split($0, arr, ":")
    for (i in arr)
        print i, arr[i]
}' file
```

### Array Examples

```bash
# Count occurrences
awk '{ count[$1]++ } END { for (word in count) print word, count[word] }' file

# Sum by category
awk '{ sum[$1] += $2 } END { for (cat in sum) print cat, sum[cat] }' file

# Find unique values
awk '{ seen[$1]++ } END { for (val in seen) print val }' file

# Group data
awk '{
    group[$1] = group[$1] " " $2
} END {
    for (key in group)
        print key ":", group[key]
}' file

# Sorted output (requires external sort)
awk '{ count[$1]++ } END { for (word in count) print count[word], word }' file | sort -rn
```

---

## Built-in Functions

### String Functions

```bash
# length(string)
awk '{ print length($1) }' file

# substr(string, start, length)
awk '{ print substr($1, 1, 3) }' file

# index(string, substring)
awk '{ print index($0, "error") }' file

# tolower(string)
awk '{ print tolower($1) }' file

# toupper(string)
awk '{ print toupper($1) }' file

# split(string, array, separator)
awk '{ split($0, arr, ":"); print arr[1] }' file

# gsub(regex, replacement, string)
awk '{ gsub(/old/, "new"); print }' file

# sub(regex, replacement, string) - first occurrence only
awk '{ sub(/old/, "new"); print }' file

# match(string, regex)
awk '{ if (match($0, /[0-9]+/)) print substr($0, RSTART, RLENGTH) }' file

# sprintf(format, ...)
awk '{ str = sprintf("%s:%d", $1, $2); print str }' file
```

### Mathematical Functions

```bash
# int(number)
awk '{ print int($1) }' file

# sqrt(number)
awk '{ print sqrt($1) }' file

# sin(number), cos(number), atan2(y, x)
awk 'BEGIN { print sin(0), cos(0), atan2(1, 1) }'

# exp(number), log(number)
awk '{ print exp($1), log($1) }' file

# rand() - random number 0-1
awk 'BEGIN { print rand() }'

# srand(seed) - seed random number generator
awk 'BEGIN { srand(); print rand() }'
```

---

## BEGIN and END Blocks

### BEGIN Block

Executed before processing any input.

```bash
# Initialize variables
awk 'BEGIN { sum = 0 } { sum += $1 } END { print sum }' file

# Print header
awk 'BEGIN { print "Name\tAge\tCity" } { print }' file

# Set field separator
awk 'BEGIN { FS=":" } { print $1 }' file

# Print formatted header
awk 'BEGIN {
    print "=============================="
    print "      Sales Report"
    print "=============================="
} { print }' file
```

### END Block

Executed after processing all input.

```bash
# Print summary
awk '{ sum += $1 } END { print "Total:", sum }' file

# Print statistics
awk '{
    sum += $1
    count++
} END {
    print "Count:", count
    print "Total:", sum
    print "Average:", sum/count
}' file

# Print footer
awk '{ print } END { print "--- End of File ---" }' file
```

### Combined BEGIN and END

```bash
awk '
BEGIN {
    print "Processing file..."
    count = 0
}
{
    count++
    sum += $1
}
END {
    print "Processed", count, "lines"
    print "Total:", sum
    print "Average:", sum/count
}
' file
```

---

## Common Patterns

### CSV Processing

```bash
# Parse CSV (simple)
awk -F',' '{ print $1, $2 }' file.csv

# Parse CSV with quoted fields
awk 'BEGIN { FPAT = "([^,]+)|(\"[^\"]+\")" } { print $1, $2 }' file.csv

# Convert CSV to TSV
awk 'BEGIN { FS=","; OFS="\t" } { print $1, $2, $3 }' file.csv

# Add CSV header
awk 'BEGIN { FS=","; OFS=","; print "Name,Age,City" } { print }' file.csv

# Skip CSV header
awk 'NR>1 { print }' file.csv

# Sum column in CSV
awk -F',' 'NR>1 { sum += $2 } END { print sum }' file.csv
```

### Log Analysis

```bash
# Count log levels
awk '{ count[$1]++ } END { for (level in count) print level, count[level] }' app.log

# Filter by date
awk '/2024-01-15/' app.log

# Extract error messages
awk '/ERROR/ { print $0 }' app.log

# Count errors by hour
awk '/ERROR/ {
    split($1, time, ":")
    hour[time[1]]++
} END {
    for (h in hour)
        print h, hour[h]
}' app.log

# Top IP addresses in access log
awk '{ ip[$1]++ } END { for (i in ip) print ip[i], i }' access.log | sort -rn | head -10

# Response time statistics
awk '{
    sum += $NF
    count++
    if ($NF > max) max = $NF
    if (min == 0 || $NF < min) min = $NF
} END {
    print "Count:", count
    print "Average:", sum/count
    print "Min:", min
    print "Max:", max
}' access.log
```

### Data Aggregation

```bash
# Sum by category
awk '{ sum[$1] += $2 } END { for (cat in sum) print cat, sum[cat] }' file

# Count by category
awk '{ count[$1]++ } END { for (cat in count) print cat, count[cat] }' file

# Average by category
awk '{
    sum[$1] += $2
    count[$1]++
} END {
    for (cat in sum)
        print cat, sum[cat]/count[cat]
}' file

# Min/Max by category
awk '{
    if (!($1 in max) || $2 > max[$1])
        max[$1] = $2
    if (!($1 in min) || $2 < min[$1])
        min[$1] = $2
} END {
    for (cat in max)
        print cat, "min:", min[cat], "max:", max[cat]
}' file
```

### Report Generation

```bash
# Simple report
awk '
BEGIN {
    print "="*40
    print "Sales Report"
    print "="*40
    printf "%-15s %10s %10s\n", "Product", "Quantity", "Revenue"
    print "-"*40
}
{
    printf "%-15s %10d %10.2f\n", $1, $2, $3
    total += $3
}
END {
    print "-"*40
    printf "%-15s %10s %10.2f\n", "TOTAL", "", total
    print "="*40
}
' sales.txt

# Formatted table
awk '
BEGIN {
    FS=","
    printf "| %-10s | %-20s | %-8s |\n", "ID", "Name", "Score"
    print "+------------+----------------------+----------+"
}
{
    printf "| %-10s | %-20s | %-8s |\n", $1, $2, $3
}
' data.csv
```

---

## Advanced Techniques

### Multiple Input Files

```bash
# Process multiple files
awk '{ print FILENAME, $0 }' file1 file2 file3

# Different action per file
awk 'FNR==1 { print "File:", FILENAME } { print }' file1 file2

# Join files by key
awk '
NR==FNR { a[$1] = $2; next }
{ print $1, $2, a[$1] }
' file1 file2

# Merge files side by side
awk '
NR==FNR { a[FNR] = $0; next }
{ print a[FNR], $0 }
' file1 file2
```

### Field Manipulation

```bash
# Swap fields
awk '{ print $2, $1 }' file

# Add new field
awk '{ print $0, $1+$2 }' file

# Remove field
awk '{ $3 = ""; print }' file

# Modify field
awk '{ $1 = toupper($1); print }' file

# Reorder fields
awk '{ print $3, $1, $2 }' file

# Print fields in reverse
awk '{ for (i=NF; i>=1; i--) printf "%s ", $i; print "" }' file
```

### External Commands

```bash
# Execute shell command
awk '{ system("echo Processing: " $1) }' file

# Get command output
awk '{ "date" | getline d; print d, $0 }' file

# Pipe to command
awk '{ print $1 | "sort" }' file

# Close pipe
awk '{
    print $1 | "sort"
} END {
    close("sort")
}' file
```

### Multi-line Records

```bash
# Paragraph mode (blank line separated)
awk 'BEGIN { RS="" } { print NR, $0 }' file

# Custom record separator
awk 'BEGIN { RS=";" } { print }' file

# Multi-line matching
awk 'BEGIN { RS="" } /pattern/' file
```

---

## Practical Examples

### System Administration

```bash
# Disk usage by user
df -h | awk 'NR>1 { print $5, $6 }' | sort -rn

# Memory usage
free -m | awk 'NR==2 { printf "Memory: %.2f%%\n", $3/$2*100 }'

# Process monitoring
ps aux | awk 'NR>1 { mem[$1] += $4 } END { for (user in mem) print user, mem[user] }'

# Extract specific processes
ps aux | awk '$11 ~ /python/ { print $2, $11 }'

# Network connections count
netstat -an | awk '/ESTABLISHED/ { count[$5]++ } END { for (ip in count) print count[ip], ip }' | sort -rn
```

### Data Transformation

```bash
# Convert spaces to tabs
awk '{ gsub(/ /, "\t"); print }' file

# Remove blank lines
awk 'NF > 0' file

# Remove duplicate lines (keeps first)
awk '!seen[$0]++' file

# Number lines
awk '{ print NR, $0 }' file

# Reverse line order
awk '{ lines[NR] = $0 } END { for (i=NR; i>0; i--) print lines[i] }' file

# Print specific columns
awk '{ print $1, $3, $5 }' file

# Column alignment
awk '{ printf "%-20s %-10s %8.2f\n", $1, $2, $3 }' file
```

### Text Processing

```bash
# Word frequency
awk '{ for (i=1; i<=NF; i++) count[$i]++ } END { for (word in count) print word, count[word] }' file

# Extract URLs
awk '{ for (i=1; i<=NF; i++) if ($i ~ /^https?:\/\//) print $i }' file

# Extract email addresses
awk '{ for (i=1; i<=NF; i++) if ($i ~ /@/) print $i }' file

# Remove HTML tags
awk '{ gsub(/<[^>]*>/, ""); print }' file.html

# Extract phone numbers
awk '/[0-9]{3}-[0-9]{3}-[0-9]{4}/ { print $0 }' file

# Line length statistics
awk '{ len = length($0); sum += len; if (len > max) max = len } END { print "Avg:", sum/NR, "Max:", max }' file
```

### Database-like Operations

```bash
# Select specific columns
awk '{ print $1, $3, $5 }' file

# Where clause
awk '$2 > 100 && $3 == "Active"' file

# Group by and sum
awk '{ sum[$1] += $2 } END { for (k in sum) print k, sum[k] }' file

# Join two files
awk 'NR==FNR { a[$1] = $2; next } { print $0, a[$1] }' lookup.txt data.txt

# Left outer join
awk 'NR==FNR { a[$1] = $2; next } { print $0, ($1 in a) ? a[$1] : "NULL" }' file1 file2
```

---

## AWK Scripts

### Using Script Files

```bash
# Create script file (script.awk)
#!/usr/bin/awk -f

BEGIN {
    print "Processing..."
}

{
    # Process each line
    sum += $1
}

END {
    print "Total:", sum
}

# Execute script
awk -f script.awk file.txt

# Make executable
chmod +x script.awk
./script.awk file.txt
```

### Complex Script Example

```awk
#!/usr/bin/awk -f

# Log analyzer script

BEGIN {
    FS = " "
    print "Log Analysis Report"
    print "==================="
}

# Count by log level
{
    level[$1]++
}

# Extract errors
$1 == "ERROR" {
    errors[NR] = $0
}

# Time-based analysis
{
    if (match($2, /([0-9]{2}):/, time)) {
        hour[time[1]]++
    }
}

END {
    # Print log levels
    print "\nLog Levels:"
    for (l in level)
        print l, level[l]

    # Print hourly distribution
    print "\nHourly Distribution:"
    for (h in hour)
        print h ":00", hour[h]

    # Print errors
    if (length(errors) > 0) {
        print "\nErrors:"
        for (line in errors)
            print errors[line]
    }
}
```

---

## Best Practices

### Performance Tips

```bash
# Use built-in variables instead of functions
awk '{ if (NF > 5) print }' file          # Fast
awk '{ if (length($0) > 50) print }' file # Slower

# Avoid unnecessary regex
awk '$1 == "error"' file                   # Fast
awk '$1 ~ /^error$/' file                  # Slower

# Exit early when possible
awk '{ if (found) exit } /pattern/ { found = 1; print }' file

# Use arrays for lookups
awk 'NR==FNR { lookup[$1] = 1; next } $1 in lookup' keys.txt data.txt
```

### Code Organization

```bash
# Use meaningful variable names
awk '{ total_sales += $3 } END { print total_sales }' file

# Comment your code
awk '
{
    # Calculate total including tax
    subtotal = $2 * $3
    tax = subtotal * 0.08
    total = subtotal + tax
    print $1, total
}
' file

# Use functions (gawk)
awk '
function celsius_to_fahrenheit(c) {
    return c * 9/5 + 32
}

{
    print $1, celsius_to_fahrenheit($2)
}
' file
```

### Error Handling

```bash
# Check for division by zero
awk '{ if ($2 != 0) print $1/$2; else print "Error: division by zero" }' file

# Validate input
awk '{ if (NF >= 3) print; else print "Invalid line:", NR > "/dev/stderr" }' file

# Check if file exists (in BEGIN)
awk 'BEGIN { if (system("test -f " ARGV[1]) != 0) { print "File not found"; exit 1 } }' file
```

---

## Common Gotchas

### Field Modification

```bash
# Modifying a field rebuilds $0
awk '{ $1 = "new"; print }' file  # Rebuilds entire line

# Fields are 1-indexed, not 0-indexed
awk '{ print $0 }' file  # Entire line
awk '{ print $1 }' file  # First field
```

### String vs Number

```bash
# AWK converts automatically
awk '{ print $1 + $2 }' file     # Numeric addition
awk '{ print $1 $2 }' file       # String concatenation

# Force numeric comparison
awk '{ if ($1 + 0 > $2 + 0) print }' file

# Force string comparison
awk '{ if ($1 "" > $2 "") print }' file
```

### Variable Scope

```bash
# Variables are global by default
awk '{
    x = $1
    y = $2
}
END {
    print x, y  # Last values from input
}' file

# Function parameters are local
awk '
function f(local_var) {
    local_var = 10
}
BEGIN {
    global_var = 5
    f(global_var)
    print global_var  # Still 5
}
'
```

---

## AWK Versions

### Differences

- **awk**: Original AT&T awk (limited features)
- **nawk**: New awk (more features)
- **gawk**: GNU awk (most features, recommended)
- **mawk**: Faster, fewer features

### GAWK-specific Features

```bash
# FPAT (field pattern)
gawk 'BEGIN { FPAT = "([^,]+)|(\"[^\"]+\")" } { print $2 }' file.csv

# Include files
gawk -f lib.awk -f script.awk file

# Two-way pipes
gawk '{ print $0 |& "sort"; "sort" |& getline x; print x }' file

# Multidimensional arrays
gawk '{ arr[$1][$2] = $3 }' file

# switch statement
gawk '{ switch ($1) { case "a": print "A"; break; case "b": print "B"; break } }' file
```

---

## Quick Reference

### Common Options

| Option | Description |
|--------|-------------|
| `-F sep` | Field separator |
| `-f file` | Read program from file |
| `-v var=val` | Set variable |
| `-W version` | Show version |

### Built-in Variables

| Variable | Description |
|----------|-------------|
| `$0` | Entire line |
| `$1, $2, ...` | Fields |
| `NF` | Number of fields |
| `NR` | Record number |
| `FNR` | File record number |
| `FS` | Field separator |
| `OFS` | Output field separator |
| `RS` | Record separator |
| `ORS` | Output record separator |
| `FILENAME` | Current filename |

### Operators

| Operator | Description |
|----------|-------------|
| `~` | Match |
| `!~` | Don't match |
| `==` | Equal |
| `!=` | Not equal |
| `<, >, <=, >=` | Comparisons |
| `&&` | AND |
| `||` | OR |
| `!` | NOT |

---

## Resources

- **GNU AWK Manual**: https://www.gnu.org/software/gawk/manual/
- **AWK Tutorial**: https://www.grymoire.com/Unix/Awk.html
- **The AWK Programming Language** (Book by Aho, Weinberger, Kernighan)

---

AWK is an incredibly powerful tool for text processing. Master these patterns and you'll be able to handle virtually any text manipulation task from the command line.
