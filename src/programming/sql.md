# SQL (Structured Query Language)

## Overview

SQL is a domain-specific language used for managing and manipulating relational databases. It's the standard language for relational database management systems (RDBMS).

**Key Concepts:**
- Declarative language (what, not how)
- ACID properties (Atomicity, Consistency, Isolation, Durability)
- Set-based operations
- Data definition, manipulation, and querying
- Transaction management

**Popular Database Systems:**
- PostgreSQL
- MySQL/MariaDB
- Oracle Database
- Microsoft SQL Server
- SQLite

---

## Basic Syntax

### Data Types

```sql
-- Numeric
INT, INTEGER              -- Whole numbers
SMALLINT, BIGINT          -- Different sizes
DECIMAL(10, 2), NUMERIC   -- Fixed-point numbers
FLOAT, REAL, DOUBLE       -- Floating-point numbers

-- String
CHAR(10)                  -- Fixed length
VARCHAR(255)              -- Variable length
TEXT                      -- Long text

-- Date and Time
DATE                      -- Date only
TIME                      -- Time only
TIMESTAMP, DATETIME       -- Date and time
YEAR                      -- Year only

-- Boolean
BOOLEAN, BOOL             -- True/False

-- Binary
BLOB                      -- Binary large object
BYTEA (PostgreSQL)        -- Binary data

-- Other
JSON, JSONB (PostgreSQL)  -- JSON data
UUID                      -- Universally unique identifier
ENUM('small', 'medium')   -- Enumerated type
```

---

## DDL (Data Definition Language)

### CREATE

```sql
-- Create database
CREATE DATABASE mydb;
CREATE DATABASE IF NOT EXISTS mydb;

-- Use database
USE mydb;

-- Create table
CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    age INT CHECK (age >= 18),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create table with foreign key
CREATE TABLE posts (
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    published BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create table with composite primary key
CREATE TABLE user_roles (
    user_id INT,
    role_id INT,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, role_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (role_id) REFERENCES roles(id)
);

-- Create table from query
CREATE TABLE archived_users AS
SELECT * FROM users WHERE created_at < '2020-01-01';
```

### ALTER

```sql
-- Add column
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- Modify column
ALTER TABLE users MODIFY COLUMN email VARCHAR(150);
ALTER TABLE users ALTER COLUMN age SET DEFAULT 18;

-- Rename column
ALTER TABLE users RENAME COLUMN username TO user_name;

-- Drop column
ALTER TABLE users DROP COLUMN phone;

-- Add constraint
ALTER TABLE users ADD CONSTRAINT chk_age CHECK (age >= 18);
ALTER TABLE users ADD UNIQUE (email);

-- Drop constraint
ALTER TABLE users DROP CONSTRAINT chk_age;

-- Rename table
ALTER TABLE users RENAME TO customers;
```

### DROP

```sql
-- Drop table
DROP TABLE users;
DROP TABLE IF EXISTS users;

-- Drop database
DROP DATABASE mydb;
DROP DATABASE IF EXISTS mydb;

-- Truncate (delete all rows, keep structure)
TRUNCATE TABLE users;
```

---

## DML (Data Manipulation Language)

### INSERT

```sql
-- Insert single row
INSERT INTO users (username, email, password, age)
VALUES ('alice', 'alice@example.com', 'hashed_pwd', 30);

-- Insert multiple rows
INSERT INTO users (username, email, password, age)
VALUES
    ('bob', 'bob@example.com', 'hashed_pwd', 25),
    ('charlie', 'charlie@example.com', 'hashed_pwd', 35),
    ('diana', 'diana@example.com', 'hashed_pwd', 28);

-- Insert from select
INSERT INTO archived_users
SELECT * FROM users WHERE created_at < '2020-01-01';

-- Insert or update (MySQL)
INSERT INTO users (id, username, email)
VALUES (1, 'alice', 'alice@example.com')
ON DUPLICATE KEY UPDATE email = VALUES(email);

-- Insert or ignore (MySQL)
INSERT IGNORE INTO users (username, email)
VALUES ('alice', 'alice@example.com');

-- Upsert (PostgreSQL)
INSERT INTO users (id, username, email)
VALUES (1, 'alice', 'alice@example.com')
ON CONFLICT (id) DO UPDATE SET email = EXCLUDED.email;
```

### UPDATE

```sql
-- Update single row
UPDATE users
SET email = 'newemail@example.com'
WHERE id = 1;

-- Update multiple columns
UPDATE users
SET email = 'alice@newdomain.com',
    age = 31,
    updated_at = CURRENT_TIMESTAMP
WHERE username = 'alice';

-- Update with condition
UPDATE users
SET age = age + 1
WHERE created_at < '2020-01-01';

-- Update from join
UPDATE users u
INNER JOIN orders o ON u.id = o.user_id
SET u.total_orders = (
    SELECT COUNT(*) FROM orders WHERE user_id = u.id
)
WHERE o.created_at > '2024-01-01';

-- Update all rows (dangerous!)
UPDATE users SET active = TRUE;
```

### DELETE

```sql
-- Delete specific row
DELETE FROM users WHERE id = 1;

-- Delete with condition
DELETE FROM users WHERE created_at < '2020-01-01';

-- Delete with join (MySQL)
DELETE u FROM users u
INNER JOIN orders o ON u.id = o.user_id
WHERE o.status = 'cancelled';

-- Delete all rows (dangerous!)
DELETE FROM users;

-- Soft delete (recommended)
UPDATE users SET deleted_at = CURRENT_TIMESTAMP WHERE id = 1;
```

---

## DQL (Data Query Language)

### SELECT

```sql
-- Select all columns
SELECT * FROM users;

-- Select specific columns
SELECT username, email FROM users;

-- Select with alias
SELECT username AS name, email AS contact FROM users;

-- Select with calculation
SELECT
    username,
    age,
    YEAR(CURRENT_DATE) - YEAR(created_at) AS years_member
FROM users;

-- Select distinct
SELECT DISTINCT age FROM users;

-- Select with limit
SELECT * FROM users LIMIT 10;
SELECT * FROM users LIMIT 10 OFFSET 20;  -- Skip first 20

-- Select top (SQL Server)
SELECT TOP 10 * FROM users;
```

### WHERE

```sql
-- Basic conditions
SELECT * FROM users WHERE age > 25;
SELECT * FROM users WHERE username = 'alice';
SELECT * FROM users WHERE age >= 18 AND age <= 65;

-- IN operator
SELECT * FROM users WHERE age IN (25, 30, 35);
SELECT * FROM users WHERE username IN ('alice', 'bob', 'charlie');

-- BETWEEN
SELECT * FROM users WHERE age BETWEEN 18 AND 65;
SELECT * FROM users WHERE created_at BETWEEN '2023-01-01' AND '2023-12-31';

-- LIKE (pattern matching)
SELECT * FROM users WHERE email LIKE '%@gmail.com';
SELECT * FROM users WHERE username LIKE 'a%';      -- Starts with 'a'
SELECT * FROM users WHERE username LIKE '%a';      -- Ends with 'a'
SELECT * FROM users WHERE username LIKE '%a%';     -- Contains 'a'
SELECT * FROM users WHERE username LIKE 'a_b';     -- _ matches single char

-- IS NULL / IS NOT NULL
SELECT * FROM users WHERE phone IS NULL;
SELECT * FROM users WHERE phone IS NOT NULL;

-- NOT
SELECT * FROM users WHERE NOT age > 25;
SELECT * FROM users WHERE age NOT IN (25, 30, 35);

-- Combining conditions
SELECT * FROM users
WHERE (age > 25 OR username LIKE 'a%')
AND email IS NOT NULL;
```

### ORDER BY

```sql
-- Sort ascending
SELECT * FROM users ORDER BY age;
SELECT * FROM users ORDER BY age ASC;

-- Sort descending
SELECT * FROM users ORDER BY age DESC;

-- Sort by multiple columns
SELECT * FROM users ORDER BY age DESC, username ASC;

-- Sort by calculated column
SELECT username, age * 2 AS double_age
FROM users
ORDER BY double_age DESC;

-- Sort with NULL handling
SELECT * FROM users ORDER BY phone NULLS FIRST;
SELECT * FROM users ORDER BY phone NULLS LAST;
```

### GROUP BY

```sql
-- Count users by age
SELECT age, COUNT(*) as count
FROM users
GROUP BY age;

-- Multiple aggregations
SELECT
    age,
    COUNT(*) as count,
    AVG(age) as avg_age,
    MIN(age) as min_age,
    MAX(age) as max_age
FROM users
GROUP BY age;

-- Group by multiple columns
SELECT
    age,
    YEAR(created_at) as year,
    COUNT(*) as count
FROM users
GROUP BY age, YEAR(created_at);

-- HAVING (filter groups)
SELECT age, COUNT(*) as count
FROM users
GROUP BY age
HAVING COUNT(*) > 5;

-- GROUP BY with ORDER BY
SELECT age, COUNT(*) as count
FROM users
GROUP BY age
HAVING COUNT(*) > 5
ORDER BY count DESC;
```

---

## Joins

```sql
-- INNER JOIN (only matching rows)
SELECT u.username, p.title
FROM users u
INNER JOIN posts p ON u.id = p.user_id;

-- LEFT JOIN (all from left, matching from right)
SELECT u.username, p.title
FROM users u
LEFT JOIN posts p ON u.id = p.user_id;

-- RIGHT JOIN (all from right, matching from left)
SELECT u.username, p.title
FROM users u
RIGHT JOIN posts p ON u.id = p.user_id;

-- FULL OUTER JOIN (all from both)
SELECT u.username, p.title
FROM users u
FULL OUTER JOIN posts p ON u.id = p.user_id;

-- CROSS JOIN (Cartesian product)
SELECT u.username, r.role_name
FROM users u
CROSS JOIN roles r;

-- Self join
SELECT
    e1.name as employee,
    e2.name as manager
FROM employees e1
LEFT JOIN employees e2 ON e1.manager_id = e2.id;

-- Multiple joins
SELECT
    u.username,
    p.title,
    c.content as comment
FROM users u
INNER JOIN posts p ON u.id = p.user_id
INNER JOIN comments c ON p.id = c.post_id;

-- Join with conditions
SELECT u.username, p.title
FROM users u
LEFT JOIN posts p ON u.id = p.user_id AND p.published = TRUE;
```

---

## Subqueries

```sql
-- Subquery in WHERE
SELECT username FROM users
WHERE id IN (
    SELECT user_id FROM orders WHERE total > 100
);

-- Subquery in SELECT
SELECT
    username,
    (SELECT COUNT(*) FROM posts WHERE user_id = users.id) as post_count
FROM users;

-- Subquery in FROM
SELECT avg_age FROM (
    SELECT AVG(age) as avg_age FROM users GROUP BY city
) as subquery;

-- Correlated subquery
SELECT username FROM users u
WHERE age > (
    SELECT AVG(age) FROM users WHERE city = u.city
);

-- EXISTS
SELECT username FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders WHERE user_id = u.id
);

-- NOT EXISTS
SELECT username FROM users u
WHERE NOT EXISTS (
    SELECT 1 FROM orders WHERE user_id = u.id
);

-- ANY / ALL
SELECT username FROM users
WHERE age > ANY (SELECT age FROM users WHERE city = 'NYC');

SELECT username FROM users
WHERE age > ALL (SELECT age FROM users WHERE city = 'NYC');
```

---

## Aggregate Functions

```sql
-- COUNT
SELECT COUNT(*) FROM users;
SELECT COUNT(DISTINCT age) FROM users;

-- SUM
SELECT SUM(total) FROM orders;

-- AVG
SELECT AVG(age) FROM users;

-- MIN / MAX
SELECT MIN(age), MAX(age) FROM users;

-- String aggregation (PostgreSQL)
SELECT STRING_AGG(username, ', ') FROM users;

-- GROUP_CONCAT (MySQL)
SELECT GROUP_CONCAT(username SEPARATOR ', ') FROM users;

-- Combined
SELECT
    COUNT(*) as total_users,
    AVG(age) as average_age,
    MIN(age) as youngest,
    MAX(age) as oldest,
    SUM(CASE WHEN age >= 18 THEN 1 ELSE 0 END) as adults
FROM users;
```

---

## Common Table Expressions (CTE)

```sql
-- Basic CTE
WITH active_users AS (
    SELECT * FROM users WHERE active = TRUE
)
SELECT * FROM active_users WHERE age > 25;

-- Multiple CTEs
WITH
    active_users AS (
        SELECT * FROM users WHERE active = TRUE
    ),
    user_posts AS (
        SELECT user_id, COUNT(*) as post_count
        FROM posts
        GROUP BY user_id
    )
SELECT
    au.username,
    up.post_count
FROM active_users au
LEFT JOIN user_posts up ON au.id = up.user_id;

-- Recursive CTE (hierarchy)
WITH RECURSIVE employee_hierarchy AS (
    -- Base case
    SELECT id, name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case
    SELECT e.id, e.name, e.manager_id, eh.level + 1
    FROM employees e
    INNER JOIN employee_hierarchy eh ON e.manager_id = eh.id
)
SELECT * FROM employee_hierarchy ORDER BY level;
```

---

## Window Functions

```sql
-- ROW_NUMBER
SELECT
    username,
    age,
    ROW_NUMBER() OVER (ORDER BY age DESC) as row_num
FROM users;

-- RANK / DENSE_RANK
SELECT
    username,
    score,
    RANK() OVER (ORDER BY score DESC) as rank,
    DENSE_RANK() OVER (ORDER BY score DESC) as dense_rank
FROM users;

-- Partition by
SELECT
    city,
    username,
    age,
    AVG(age) OVER (PARTITION BY city) as avg_city_age
FROM users;

-- Running total
SELECT
    date,
    amount,
    SUM(amount) OVER (ORDER BY date) as running_total
FROM sales;

-- LAG / LEAD (previous/next row)
SELECT
    date,
    revenue,
    LAG(revenue) OVER (ORDER BY date) as prev_revenue,
    LEAD(revenue) OVER (ORDER BY date) as next_revenue
FROM daily_sales;

-- NTILE (divide into buckets)
SELECT
    username,
    score,
    NTILE(4) OVER (ORDER BY score DESC) as quartile
FROM users;
```

---

## Indexes

```sql
-- Create index
CREATE INDEX idx_users_email ON users(email);

-- Create unique index
CREATE UNIQUE INDEX idx_users_username ON users(username);

-- Create composite index
CREATE INDEX idx_users_age_city ON users(age, city);

-- Create partial index (PostgreSQL)
CREATE INDEX idx_active_users ON users(username)
WHERE active = TRUE;

-- Create index with condition (filtered index - SQL Server)
CREATE INDEX idx_active_users ON users(username)
WHERE active = 1;

-- Full-text index (MySQL)
CREATE FULLTEXT INDEX idx_posts_content ON posts(title, content);

-- Drop index
DROP INDEX idx_users_email ON users;

-- Show indexes
SHOW INDEX FROM users;  -- MySQL
SELECT * FROM pg_indexes WHERE tablename = 'users';  -- PostgreSQL
```

---

## Transactions

```sql
-- Start transaction
BEGIN;
START TRANSACTION;

-- Commit transaction
COMMIT;

-- Rollback transaction
ROLLBACK;

-- Example transaction
BEGIN;

UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- Check if everything is okay
IF (SELECT balance FROM accounts WHERE id = 1) >= 0 THEN
    COMMIT;
ELSE
    ROLLBACK;
END IF;

-- Savepoint
BEGIN;
UPDATE users SET age = 30 WHERE id = 1;
SAVEPOINT my_savepoint;
UPDATE users SET age = 40 WHERE id = 2;
ROLLBACK TO SAVEPOINT my_savepoint;  -- Only rollback second update
COMMIT;

-- Transaction isolation levels
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

---

## Views

```sql
-- Create view
CREATE VIEW active_users AS
SELECT id, username, email
FROM users
WHERE active = TRUE;

-- Use view
SELECT * FROM active_users;

-- Create or replace view
CREATE OR REPLACE VIEW user_stats AS
SELECT
    u.id,
    u.username,
    COUNT(p.id) as post_count,
    COUNT(DISTINCT c.id) as comment_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
LEFT JOIN comments c ON u.id = c.user_id
GROUP BY u.id, u.username;

-- Materialized view (PostgreSQL)
CREATE MATERIALIZED VIEW user_stats_mv AS
SELECT
    u.id,
    COUNT(p.id) as post_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
GROUP BY u.id;

-- Refresh materialized view
REFRESH MATERIALIZED VIEW user_stats_mv;

-- Drop view
DROP VIEW active_users;
DROP MATERIALIZED VIEW user_stats_mv;
```

---

## Stored Procedures and Functions

```sql
-- MySQL stored procedure
DELIMITER //
CREATE PROCEDURE GetUsersByAge(IN min_age INT)
BEGIN
    SELECT * FROM users WHERE age >= min_age;
END //
DELIMITER ;

-- Call procedure
CALL GetUsersByAge(25);

-- Function (MySQL)
DELIMITER //
CREATE FUNCTION CalculateAge(birth_date DATE)
RETURNS INT
DETERMINISTIC
BEGIN
    RETURN YEAR(CURRENT_DATE) - YEAR(birth_date);
END //
DELIMITER ;

-- Use function
SELECT username, CalculateAge(birth_date) as age FROM users;

-- PostgreSQL function
CREATE OR REPLACE FUNCTION get_user_count()
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM users);
END;
$$ LANGUAGE plpgsql;

-- Call function
SELECT get_user_count();

-- Drop procedure/function
DROP PROCEDURE GetUsersByAge;
DROP FUNCTION CalculateAge;
```

---

## Common Patterns

### Pagination

```sql
-- Offset pagination
SELECT * FROM users
ORDER BY id
LIMIT 10 OFFSET 20;  -- Page 3 (0-based)

-- Cursor-based pagination (more efficient)
SELECT * FROM users
WHERE id > 100  -- Last seen ID
ORDER BY id
LIMIT 10;
```

### Finding Duplicates

```sql
-- Find duplicate emails
SELECT email, COUNT(*) as count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- Get duplicate rows with details
SELECT u.*
FROM users u
INNER JOIN (
    SELECT email FROM users
    GROUP BY email
    HAVING COUNT(*) > 1
) dup ON u.email = dup.email;
```

### Ranking

```sql
-- Top N per group
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY category ORDER BY score DESC) as rn
    FROM products
)
SELECT * FROM ranked WHERE rn <= 3;
```

### Running Totals

```sql
-- Running total
SELECT
    date,
    revenue,
    SUM(revenue) OVER (ORDER BY date) as cumulative_revenue
FROM sales
ORDER BY date;
```

### Pivot Table

```sql
-- MySQL
SELECT
    username,
    SUM(CASE WHEN YEAR(created_at) = 2023 THEN 1 ELSE 0 END) as year_2023,
    SUM(CASE WHEN YEAR(created_at) = 2024 THEN 1 ELSE 0 END) as year_2024
FROM users
GROUP BY username;

-- PostgreSQL (crosstab)
SELECT * FROM crosstab(
    'SELECT username, YEAR(created_at), COUNT(*) FROM users GROUP BY 1, 2',
    'SELECT DISTINCT YEAR(created_at) FROM users ORDER BY 1'
) AS ct(username TEXT, year_2023 INT, year_2024 INT);
```

---

## Performance Optimization

### Best Practices

1. **Use indexes wisely**
   ```sql
   -- Index columns used in WHERE, JOIN, ORDER BY
   CREATE INDEX idx_users_email ON users(email);
   ```

2. **Avoid SELECT ***
   ```sql
   -- Bad
   SELECT * FROM users;

   -- Good
   SELECT id, username, email FROM users;
   ```

3. **Use LIMIT**
   ```sql
   SELECT * FROM users LIMIT 100;
   ```

4. **Use JOIN instead of subqueries when possible**
   ```sql
   -- Slower
   SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);

   -- Faster
   SELECT DISTINCT u.* FROM users u INNER JOIN orders o ON u.id = o.user_id;
   ```

5. **Use EXPLAIN to analyze queries**
   ```sql
   EXPLAIN SELECT * FROM users WHERE email = 'alice@example.com';
   EXPLAIN ANALYZE SELECT * FROM users WHERE age > 25;
   ```

6. **Avoid functions on indexed columns**
   ```sql
   -- Bad (can't use index)
   SELECT * FROM users WHERE YEAR(created_at) = 2024;

   -- Good
   SELECT * FROM users
   WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
   ```

7. **Use covering indexes**
   ```sql
   CREATE INDEX idx_users_email_username ON users(email, username);
   -- This query can be satisfied entirely from the index
   SELECT username FROM users WHERE email = 'alice@example.com';
   ```

---

## Common Functions

### String Functions

```sql
-- CONCAT
SELECT CONCAT(first_name, ' ', last_name) as full_name FROM users;

-- UPPER / LOWER
SELECT UPPER(username), LOWER(email) FROM users;

-- LENGTH / CHAR_LENGTH
SELECT LENGTH(username), CHAR_LENGTH(username) FROM users;

-- SUBSTRING
SELECT SUBSTRING(email, 1, 10) FROM users;

-- TRIM
SELECT TRIM(username) FROM users;

-- REPLACE
SELECT REPLACE(email, '@gmail.com', '@newdomain.com') FROM users;
```

### Date Functions

```sql
-- Current date/time
SELECT NOW(), CURRENT_DATE, CURRENT_TIME;

-- Date arithmetic
SELECT DATE_ADD(created_at, INTERVAL 30 DAY) FROM users;
SELECT DATE_SUB(created_at, INTERVAL 1 YEAR) FROM users;

-- Date difference
SELECT DATEDIFF(NOW(), created_at) as days_since_creation FROM users;

-- Date formatting
SELECT DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s') FROM users;

-- Extract parts
SELECT
    YEAR(created_at) as year,
    MONTH(created_at) as month,
    DAY(created_at) as day
FROM users;
```

### Conditional Functions

```sql
-- CASE
SELECT
    username,
    CASE
        WHEN age < 18 THEN 'Minor'
        WHEN age >= 18 AND age < 65 THEN 'Adult'
        ELSE 'Senior'
    END as age_group
FROM users;

-- IF (MySQL)
SELECT IF(age >= 18, 'Adult', 'Minor') as status FROM users;

-- COALESCE (first non-null value)
SELECT COALESCE(phone, email, 'No contact') FROM users;

-- NULLIF (return NULL if equal)
SELECT NULLIF(age, 0) FROM users;
```

---

## Security Best Practices

1. **Use parameterized queries** (prevent SQL injection)
   ```sql
   -- Bad (vulnerable to SQL injection)
   SELECT * FROM users WHERE username = '$user_input';

   -- Good (parameterized)
   SELECT * FROM users WHERE username = ?;
   ```

2. **Principle of least privilege**
   - Grant minimum necessary permissions
   - Use separate accounts for different applications

3. **Encrypt sensitive data**
   ```sql
   -- Store password hashes, never plain text
   INSERT INTO users (username, password_hash)
   VALUES ('alice', SHA2('password', 256));
   ```

4. **Regular backups**
   ```bash
   mysqldump -u root -p database_name > backup.sql
   pg_dump database_name > backup.sql
   ```

5. **Input validation**
   - Validate and sanitize all user inputs
   - Use constraints in database schema

---

## Database-Specific Features

### PostgreSQL

```sql
-- Array type
CREATE TABLE users (tags TEXT[]);
INSERT INTO users (tags) VALUES (ARRAY['admin', 'moderator']);
SELECT * FROM users WHERE 'admin' = ANY(tags);

-- JSON type
CREATE TABLE users (metadata JSONB);
INSERT INTO users (metadata) VALUES ('{"age": 30, "city": "NYC"}');
SELECT metadata->>'age' FROM users;

-- Generate series
SELECT * FROM generate_series(1, 10);
```

### MySQL

```sql
-- Auto increment
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY
);

-- Full-text search
CREATE FULLTEXT INDEX ft_content ON posts(content);
SELECT * FROM posts WHERE MATCH(content) AGAINST('search term');

-- JSON functions
SELECT JSON_EXTRACT(metadata, '$.age') FROM users;
```

---

## Common Database Tools

- **MySQL Workbench**: GUI for MySQL
- **pgAdmin**: GUI for PostgreSQL
- **DBeaver**: Universal database tool
- **TablePlus**: Modern database client
- **DataGrip**: JetBrains database IDE

---

## Command Line Tools

```bash
# MySQL
mysql -u root -p
mysql -u root -p database_name < backup.sql
mysqldump -u root -p database_name > backup.sql

# PostgreSQL
psql -U postgres
psql -U postgres database_name < backup.sql
pg_dump database_name > backup.sql

# SQLite
sqlite3 database.db
.tables
.schema table_name
.quit
```
