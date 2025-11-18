# Testing & Quality

Testing strategies, frameworks, and best practices for ensuring code quality and reliability.

## Topics Covered

- **[Unit Testing](unit_testing.md)**: Testing individual functions and classes in isolation
- **[Integration Testing](integration.md)**: Testing component interactions and APIs
- **[E2E Testing](e2e_testing.md)**: End-to-end testing of complete user workflows
- **[pytest](pytest.md)**: Python testing framework with fixtures and parametrization
- **[TDD](tdd.md)**: Test-driven development approaches and best practices
- **Test Frameworks**: pytest, Jest, unittest
- **Mocking**: Isolating code under test
- **Coverage**: Measuring test completeness
- **Debugging**: Finding and fixing issues
- **Code Quality**: Linting, formatting, static analysis
- **Performance Testing**: Load and stress testing

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

1. **Fast**: Tests run quickly
2. **Independent**: No dependencies between tests
3. **Repeatable**: Consistent results
4. **Self-checking**: Pass/fail obvious
5. **Timely**: Written with code

## Navigation

Learn strategies to build reliable, quality software.
