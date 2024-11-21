# grep

grep is a command-line utility for searching for text in files. It is a powerful tool that can be used to search for text in files, directories, and more.

## Commonly Used `grep` Commands

1. **Search for a specific string in a file:**
   ```bash
   grep "search_string" filename
   ```

2. **Search for a string in multiple files:**
   ```bash
   grep "search_string" file1 file2 file3
   ```

3. **Search recursively in directories:**
   ```bash
   grep -r "search_string" /path/to/directory
   ```

4. **Search for a string ignoring case:**
   ```bash
   grep -i "search_string" filename
   ```

5. **Search for a whole word:**
   ```bash
   grep -w "word" filename
   ```

6. **Search for a string and display line numbers:**
   ```bash
   grep -n "search_string" filename
   ```

7. **Search for a string and display count of matching lines:**
   ```bash
   grep -c "search_string" filename
   ```

8. **Search for a string and display only matching part:**
   ```bash
   grep -o "search_string" filename
   ```

9. **Search for lines that do not match the string:**
   ```bash
   grep -v "search_string" filename
   ```

10. **Search for multiple patterns:**
    ```bash
    grep -e "pattern1" -e "pattern2" filename
    ```

11. **Search for a string in compressed files:**
    ```bash
    zgrep "search_string" compressed_file.gz
    ```

12. **Search for a string and display context lines:**
    ```bash
    grep -C 3 "search_string" filename
    ```

These commands cover a variety of common use cases for the `grep` command, making it a versatile tool for text searching and manipulation.
