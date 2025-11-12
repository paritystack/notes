# Unit Testing

## Overview

Unit testing verifies individual functions or classes work correctly in isolation.

## Python - pytest

```python
# test_calculator.py
import pytest
from calculator import add, divide

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_add_floats():
    assert add(0.1, 0.2) == pytest.approx(0.3)

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)

@pytest.fixture
def calculator():
    """Setup fixture"""
    return Calculator()

def test_with_fixture(calculator):
    assert calculator.add(2, 3) == 5

@pytest.mark.parametrize("x,y,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add_multiple(x, y, expected):
    assert add(x, y) == expected
```

## JavaScript - Jest

```javascript
// calculator.test.js
describe('Calculator', () => {
  test('add function', () => {
    expect(add(2, 3)).toBe(5);
  });

  test('divide by zero throws error', () => {
    expect(() => divide(10, 0)).toThrow();
  });

  test('floating point', () => {
    expect(add(0.1, 0.2)).toBeCloseTo(0.3);
  });
});
```

## Java - JUnit

```java
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class CalculatorTest {
    private Calculator calc = new Calculator();

    @Test
    void testAdd() {
        assertEquals(5, calc.add(2, 3));
    }

    @Test
    void testDivideByZero() {
        assertThrows(ArithmeticException.class,
            () -> calc.divide(10, 0));
    }

    @BeforeEach
    void setup() {
        calc = new Calculator();
    }
}
```

## Mocking

Isolate code under test:

```python
from unittest.mock import Mock, patch

@patch('module.external_api')
def test_with_mock(mock_api):
    mock_api.return_value = {"status": "ok"}
    result = my_function()
    assert result == "success"
    mock_api.assert_called_once()
```

## Best Practices

1. **One assertion per test** (or related)
2. **Arrange-Act-Assert pattern**
3. **Descriptive names**: `test_add_positive_numbers`
4. **Test behavior, not implementation**
5. **Test edge cases and errors**

## Coverage

```bash
# Python coverage
pytest --cov=myapp tests/

# JavaScript coverage
npm test -- --coverage
```

**Target**: 80%+ coverage

## Common Assertions

```python
assert x == y           # Equality
assert x > y            # Comparison
assert x is None        # Identity
assert x in list        # Membership
with raises(Exception): # Exception
    function()
```

## ELI10

Unit tests are like checking individual pieces:
- Test each part separately
- Make sure it works alone
- Catch problems early!

Like quality control in a factory!

## Further Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Jest Documentation](https://jestjs.io/)
- [JUnit 5 Guide](https://junit.org/junit5/docs/current/user-guide/)
