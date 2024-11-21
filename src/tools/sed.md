# sed

sed is a stream editor that allows you to manipulate text files. It is a powerful tool that can be used to search for and replace text, delete text, and more.


## Commonly Used `sed` Commands

1. **Replace all occurrences of a string in a file:**
   ```bash
   sed -i 's/old_string/new_string/g' filename
   ```

2. **Delete lines containing a specific pattern:**
   ```bash
   sed -i '/pattern/d' filename
   ```

3. **Print only lines that match a pattern:**
   ```bash
   sed -n '/pattern/p' filename
   ```

4. **Insert a line after a match:**
   ```bash
   sed -i '/pattern/a\new_line' filename
   ```

5. **Insert a line before a match:**
   ```bash
   sed -i '/pattern/i\new_line' filename
   ```

6. **Replace text on a specific line number:**
   ```bash
   sed -i '3s/old_text/new_text/' filename
   ```

7. **Delete a specific line number:**
   ```bash
   sed -i '5d' filename
   ```

8. **Replace text between two patterns:**
   ```bash
   sed -i '/start_pattern/,/end_pattern/s/old_text/new_text/g' filename
   ```

9. **Print lines between two patterns:**
   ```bash
   sed -n '/start_pattern/,/end_pattern/p' filename
   ```

10. **Append text to the end of each line:**
    ```bash
    sed -i 's/$/append_text/' filename
    ```

These commands cover a variety of common use cases for the `sed` command, making it a versatile tool for text manipulation and processing.
