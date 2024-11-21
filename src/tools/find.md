# find

find is a command-line utility for searching for files in directories. It is a powerful tool that can be used to search for files in directories, subdirectories, and more.


## Commonly Used `find` Commands

1. **Find files by name:**
   ```bash
   find /path/to/directory -name "filename"
   find . -name "*.py"
   ```

2. **Find files by extension:**
   ```bash
   find /path/to/directory -name "*.ext"
   ```

3. **Find files by type (e.g., directories):**
   ```bash
   find /path/to/directory -type d
   ```

4. **Find files by size (e.g., files larger than 100MB):**
   ```bash
   find /path/to/directory -size +100M
   ```

5. **Find files modified in the last 7 days:**
   ```bash
   find /path/to/directory -mtime -7
   ```

6. **Find files accessed in the last 7 days:**
   ```bash
   find /path/to/directory -atime -7
   ```

7. **Find files and execute a command on them (e.g., delete):**
   ```bash
   find /path/to/directory -name "*.tmp" -exec rm {} \;
   ```

8. **Find files by permissions (e.g., files with 777 permissions):**
   ```bash
   find /path/to/directory -perm 777
   ```

9. **Find empty files and directories:**
   ```bash
   find /path/to/directory -empty
   ```

10. **Find files by user:**
    ```bash
    find /path/to/directory -user username
    ```

11. **Find files by group:**
    ```bash
    find /path/to/directory -group groupname
    ```

12. **Find files excluding a specific path:**
    ```bash
    find /path/to/directory -path /exclude/path -prune -o -name "*.ext" -print
    ```

These commands cover a variety of common use cases for the `find` command, making it a versatile tool for file searching and manipulation.
