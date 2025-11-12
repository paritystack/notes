# Test-Driven Development (TDD)

## Overview

TDD: Write tests BEFORE writing code. Red → Green → Refactor cycle.

## Red-Green-Refactor Cycle

### 1. Red: Write Failing Test
```python
def test_add_positive_numbers():
    assert add(2, 3) == 5
```

### 2. Green: Write Minimal Code
```python
def add(a, b):
    return 5  # Hardcoded to pass test
```

### 3. Refactor: Improve Code
```python
def add(a, b):
    return a + b  # Proper implementation
```

## Benefits

✅ Better design (code written to be testable)
✅ Fewer bugs (test before shipping)
✅ Confidence (safe to refactor)
✅ Documentation (tests show usage)
✅ Less debugging (catch issues early)

## Example: Calculator

### Step 1: Red
```python
class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5
```

### Step 2: Green
```python
class Calculator:
    def add(self, a, b):
        return a + b
```

### Step 3: Refactor
```python
class Calculator:
    def add(self, a, b):
        """Add two numbers"""
        if not isinstance(a, (int, float)):
            raise TypeError("a must be number")
        return a + b
```

## TDD Best Practices

1. **Start simple**: Test one behavior
2. **One assertion** per test (usually)
3. **Clear names**: `test_add_positive_numbers`
4. **Arrange-Act-Assert**
```python
def test_withdraw():
    # Arrange
    account = Account(1000)
    # Act
    account.withdraw(200)
    # Assert
    assert account.balance == 800
```

5. **Don't skip red phase**: Ensures test can fail

## Working Test Example

```python
# calculator.py - EMPTY (start)

# test_calculator.py
def test_multiply():
    # Test fails: function doesn't exist (RED)
    assert multiply(3, 4) == 12
```

```python
# calculator.py - implement
def multiply(a, b):
    return a * b
```

```python
# test passes (GREEN)

# Refactor if needed
```

## Anti-patterns

❌ Writing all tests at once
❌ Over-engineering the implementation
❌ Ignoring red phase
❌ Poorly named tests
❌ Testing implementation, not behavior

## Coverage with TDD

TDD naturally leads to high coverage:

```python
# Typical TDD: 90%+ coverage
# Non-TDD: 20-40% coverage
```

## TDD vs BDD

**TDD**: Tests focus on unit behavior
```python
test_add_positive_numbers()
test_add_negative_numbers()
```

**BDD**: Tests focus on business behavior
```python
test_user_can_withdraw_money()
test_system_prevents_overdraft()
```

## Tools

- **pytest**: Python testing
- **Jest**: JavaScript testing
- **JUnit**: Java testing
- **RSpec**: Ruby testing

## ELI10

TDD is like building with blueprints:
1. Draw blueprint (write test)
2. Build to match (write code)
3. Improve design (refactor)

Never start building without a plan!

## Further Resources

- [TDD by Example - Kent Beck](https://www.amazon.com/Test-Driven-Development-Kent-Beck/dp/0321146530)
- [What is TDD](https://testdriven.io/blog/what-is-tdd/)
