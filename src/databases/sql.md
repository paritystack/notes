# SQL (Structured Query Language)

## Overview

SQL is the standard language for querying and managing relational databases. Used by PostgreSQL, MySQL, SQL Server, Oracle, and others.

## Basic Queries

### SELECT
```sql
SELECT column1, column2 FROM table WHERE condition;
SELECT * FROM users WHERE age > 18;
SELECT DISTINCT city FROM customers;
```

### INSERT
```sql
INSERT INTO users (name, email) VALUES ('John', 'john@example.com');
INSERT INTO users VALUES (1, 'John', 'john@example.com');
```

### UPDATE
```sql
UPDATE users SET age = 30 WHERE id = 1;
UPDATE products SET price = price * 1.1;
```

### DELETE
```sql
DELETE FROM users WHERE id = 1;
DELETE FROM logs WHERE created_at < '2023-01-01';
```

## Joins

```sql
-- INNER JOIN: Only matching rows
SELECT u.name, o.order_id
FROM users u INNER JOIN orders o ON u.id = o.user_id;

-- LEFT JOIN: All from left table
SELECT u.name, o.order_id
FROM users u LEFT JOIN orders o ON u.id = o.user_id;

-- RIGHT JOIN: All from right table
SELECT u.name, o.order_id
FROM users u RIGHT JOIN orders o ON u.id = o.user_id;

-- FULL OUTER JOIN: All rows
SELECT u.name, o.order_id
FROM users u FULL OUTER JOIN orders o ON u.id = o.user_id;
```

## Aggregation

```sql
SELECT COUNT(*) FROM users;
SELECT AVG(price) FROM products;
SELECT SUM(amount) FROM transactions WHERE status = 'completed';
SELECT MAX(salary) FROM employees;

-- GROUP BY
SELECT department, COUNT(*) FROM employees GROUP BY department;
SELECT category, AVG(price) FROM products GROUP BY category;

-- HAVING (filter groups)
SELECT department, AVG(salary)
FROM employees
GROUP BY department
HAVING AVG(salary) > 50000;
```

## Indexes

```sql
-- Create index for faster queries
CREATE INDEX idx_email ON users(email);
CREATE INDEX idx_user_date ON orders(user_id, created_at);

-- Drop index
DROP INDEX idx_email;
```

## Transactions

```sql
BEGIN TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

COMMIT;  -- Save changes
-- ROLLBACK;  -- Undo changes
```

## Window Functions

```sql
-- Rank rows
SELECT name, salary,
  RANK() OVER (ORDER BY salary DESC) as rank
FROM employees;

-- Running total
SELECT date, amount,
  SUM(amount) OVER (ORDER BY date) as running_total
FROM transactions;
```

## Common Patterns

### Duplicate Finding
```sql
SELECT email, COUNT(*) FROM users GROUP BY email HAVING COUNT(*) > 1;
```

### Top N per Group
```sql
SELECT DISTINCT ON (department) name, salary, department
FROM employees ORDER BY department, salary DESC;
```

### Data Validation
```sql
SELECT * FROM users WHERE email NOT LIKE '%@%.%';
```

## Performance Tips

1. **Use indexes** on frequently queried columns
2. **EXPLAIN** query plans: `EXPLAIN SELECT ...`
3. **Avoid SELECT *** - specify columns needed
4. **Use LIMIT** for large result sets
5. **Batch operations** instead of individual queries

## ACID Properties

- **Atomicity**: All or nothing
- **Consistency**: Valid state to valid state
- **Isolation**: Concurrent transactions independent
- **Durability**: Committed data survives failures

## ELI10

SQL is like a filing system for data:
- **SELECT**: "Show me these files"
- **INSERT**: "Add new file"
- **UPDATE**: "Modify existing file"
- **DELETE**: "Remove file"

Joins = combining data from multiple filing cabinets!

## Further Resources

- [SQL Tutorial](https://www.w3schools.com/sql/)
- [PostgreSQL Docs](https://www.postgresql.org/docs/)
- [Explain Plans](https://www.postgresql.org/docs/current/sql-explain.html)
