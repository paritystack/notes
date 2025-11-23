# Testing & Quality

Testing strategies, frameworks, and best practices for ensuring code quality and reliability.

## Topics Covered

### Testing Types

- **[Unit Testing](unit_testing.md)**: Testing individual functions and classes in isolation
- **[Integration Testing](integration.md)**: Testing component interactions and APIs
- **[E2E Testing](e2e_testing.md)**: End-to-end testing of complete user workflows

### Testing Frameworks

- **[pytest](pytest.md)**: Python testing framework with fixtures and parametrization
- **[Jest](jest.md)**: JavaScript testing framework with built-in mocking and assertions

### Testing Practices

- **[TDD](tdd.md)**: Test-driven development approaches and best practices
- **[Mocking](mocking.md)**: Isolating code under test with mocks, stubs, spies, and fakes
- **[Coverage](coverage.md)**: Measuring test completeness and setting coverage thresholds
- **[Debugging](debugging.md)**: Finding and fixing test failures and flaky tests

### Code Quality

- **[Code Quality](code_quality.md)**: Linting, formatting, type checking, and static analysis
- **[Performance Testing](performance_testing.md)**: Load testing, stress testing, and benchmarking

## Testing Pyramid

```
       E2E Tests (few)
      Integration Tests (more)
    Unit Tests (many)
```

Ratio: 70% unit, 20% integration, 10% e2e

## Test Types

- **Unit**: Individual functions/classes
- **Integration**: Multiple components together
- **End-to-End**: Full user workflows
- **Performance**: Load and speed
- **Security**: Vulnerability detection

## Best Practices

1. **Fast**: Tests run quickly to encourage frequent execution
2. **Independent**: No dependencies between tests - each can run in isolation
3. **Repeatable**: Consistent results regardless of environment or order
4. **Self-checking**: Pass/fail criteria are clear and automated
5. **Timely**: Written alongside or before production code (TDD)
6. **Maintainable**: Clear test names and simple, focused test cases
7. **Comprehensive**: Cover edge cases, error scenarios, and happy paths

## Quick Start

### Running Tests

```bash
# Python
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest --cov=myapp tests/       # With coverage

# JavaScript
npm test                        # Run all tests
npm test -- --coverage          # With coverage
npm test -- --watch             # Watch mode
```

### Writing Your First Test

**Python:**
```python
# test_calculator.py
def test_add():
    assert add(2, 3) == 5
```

**JavaScript:**
```javascript
// calculator.test.js
test('adds numbers', () => {
  expect(add(2, 3)).toBe(5);
});
```

## Testing Resources

- [Unit Testing Guide](unit_testing.md) - Start here for individual component testing
- [TDD Guide](tdd.md) - Learn test-driven development workflow
- [Mocking Guide](mocking.md) - Master test isolation techniques
- [Debugging Guide](debugging.md) - Fix failing tests efficiently

## Navigation

Explore comprehensive testing strategies, frameworks, and tools to build reliable, high-quality software with confidence.
