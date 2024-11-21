# awk

awk is a powerful tool for processing text files. It is a powerful tool that can be used to process text files, extract data, and more.


## Commonly Used `awk` Commands

1. **Print all lines in a file:**
   ```bash
   awk '{print}' filename
   ```

2. **Print the first column of a file:**
   ```bash
   awk '{print $1}' filename
   ```

3. **Print the first and second columns of a file:**
   ```bash
   awk '{print $1, $2}' filename
   ```

4. **Print lines that match a pattern:**
   ```bash
   awk '/pattern/ {print}' filename
   ```

5. **Print lines where the value in the first column is greater than 10:**
   ```bash
   awk '$1 > 10' filename
   ```

6. **Calculate the sum of values in the first column:**
   ```bash
   awk '{sum += $1} END {print sum}' filename
   ```

7. **Calculate the average of values in the first column:**
   ```bash
   awk '{sum += $1; count++} END {print sum/count}' filename
   ```

8. **Print the number of lines in a file:**
   ```bash
   awk 'END {print NR}' filename
   ```

9. **Print lines with more than 3 fields:**
   ```bash
   awk 'NF > 3' filename
   ```

10. **Replace a string in a file:**
    ```bash
    awk '{gsub(/old_string/, "new_string"); print}' filename
    ```

These commands cover a variety of common use cases for the `awk` command, making it a versatile tool for text processing and data extraction.
