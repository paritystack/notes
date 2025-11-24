# Jest

Jest is a delightful JavaScript testing framework with a focus on simplicity. It works out of the box for most JavaScript projects and provides a complete testing solution.

## Installation

```bash
# npm
npm install --save-dev jest

# yarn
yarn add --dev jest

# For TypeScript
npm install --save-dev @types/jest ts-jest

# For React
npm install --save-dev @testing-library/react @testing-library/jest-dom
```

## Basic Setup

**package.json:**
```json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage"
  }
}
```

**jest.config.js:**
```javascript
module.exports = {
  testEnvironment: 'node',
  coverageDirectory: 'coverage',
  collectCoverageFrom: [
    'src/**/*.{js,jsx}',
    '!src/**/*.test.{js,jsx}'
  ],
  testMatch: [
    '**/__tests__/**/*.[jt]s?(x)',
    '**/?(*.)+(spec|test).[jt]s?(x)'
  ]
};
```

## Writing Tests

### Basic Test Structure

```javascript
// sum.js
function sum(a, b) {
  return a + b;
}
module.exports = sum;

// sum.test.js
const sum = require('./sum');

describe('sum function', () => {
  test('adds 1 + 2 to equal 3', () => {
    expect(sum(1, 2)).toBe(3);
  });

  test('adds negative numbers', () => {
    expect(sum(-1, -2)).toBe(-3);
  });

  it('adds floating point numbers', () => {
    expect(sum(0.1, 0.2)).toBeCloseTo(0.3);
  });
});
```

### Matchers

```javascript
// Equality
expect(value).toBe(4);              // Strict equality (===)
expect(value).toEqual({ a: 1 });    // Deep equality
expect(value).not.toBe(5);          // Negation

// Truthiness
expect(value).toBeTruthy();
expect(value).toBeFalsy();
expect(value).toBeNull();
expect(value).toBeUndefined();
expect(value).toBeDefined();

// Numbers
expect(value).toBeGreaterThan(3);
expect(value).toBeGreaterThanOrEqual(3.5);
expect(value).toBeLessThan(5);
expect(value).toBeLessThanOrEqual(4.5);
expect(0.1 + 0.2).toBeCloseTo(0.3); // Floating point

// Strings
expect('team').not.toMatch(/I/);
expect('Christoph').toMatch(/stop/);

// Arrays and iterables
expect(['apple', 'banana']).toContain('apple');
expect(new Set(['apple'])).toContain('apple');

// Objects
expect(obj).toHaveProperty('name');
expect(obj).toHaveProperty('age', 25);
expect(obj).toMatchObject({ name: 'John' });

// Exceptions
expect(() => {
  throw new Error('error');
}).toThrow();
expect(() => func()).toThrow('error message');
expect(() => func()).toThrow(Error);

// Promises
await expect(promise).resolves.toBe(value);
await expect(promise).rejects.toThrow(error);
```

## Setup and Teardown

```javascript
// Before/After Each Test
describe('database tests', () => {
  beforeEach(() => {
    // Runs before each test
    initializeDatabase();
  });

  afterEach(() => {
    // Runs after each test
    clearDatabase();
  });

  test('user creation', () => {
    // Database is initialized
    const user = createUser();
    expect(user).toBeDefined();
    // Database will be cleared after
  });
});

// Before/After All Tests
describe('suite', () => {
  beforeAll(() => {
    // Runs once before all tests
    return setupDatabase();
  });

  afterAll(() => {
    // Runs once after all tests
    return teardownDatabase();
  });

  test('test 1', () => {});
  test('test 2', () => {});
});

// Scoped setup
describe('outer', () => {
  beforeEach(() => {
    console.log('outer beforeEach');
  });

  describe('inner', () => {
    beforeEach(() => {
      console.log('inner beforeEach');
    });

    test('test', () => {
      // Both beforeEach run: outer, then inner
    });
  });
});
```

## Async Testing

### Callbacks

```javascript
test('callback test', (done) => {
  function callback(data) {
    try {
      expect(data).toBe('peanut butter');
      done();
    } catch (error) {
      done(error);
    }
  }

  fetchData(callback);
});
```

### Promises

```javascript
test('promise test', () => {
  return fetchData().then(data => {
    expect(data).toBe('peanut butter');
  });
});

test('promise rejection', () => {
  return expect(fetchData()).rejects.toThrow('error');
});
```

### Async/Await

```javascript
test('async/await test', async () => {
  const data = await fetchData();
  expect(data).toBe('peanut butter');
});

test('async/await error', async () => {
  expect.assertions(1);
  try {
    await fetchData();
  } catch (error) {
    expect(error).toMatch('error');
  }
});

// Cleaner with resolves/rejects
test('async with resolves', async () => {
  await expect(fetchData()).resolves.toBe('peanut butter');
});

test('async with rejects', async () => {
  await expect(fetchData()).rejects.toThrow();
});
```

## Mocking

### Mock Functions

```javascript
// Create mock function
const mockFn = jest.fn();

// Mock with return value
mockFn.mockReturnValue(42);
expect(mockFn()).toBe(42);

// Mock with return value once
mockFn.mockReturnValueOnce(1).mockReturnValueOnce(2);
expect(mockFn()).toBe(1);
expect(mockFn()).toBe(2);

// Mock implementation
const mockCallback = jest.fn(x => x * 2);
expect(mockCallback(2)).toBe(4);

// Check mock calls
mockFn('arg1', 'arg2');
expect(mockFn).toHaveBeenCalled();
expect(mockFn).toHaveBeenCalledTimes(1);
expect(mockFn).toHaveBeenCalledWith('arg1', 'arg2');
expect(mockFn).toHaveBeenLastCalledWith('arg1', 'arg2');

// Access mock data
expect(mockFn.mock.calls.length).toBe(1);
expect(mockFn.mock.calls[0][0]).toBe('arg1');
expect(mockFn.mock.results[0].value).toBe(returnValue);
```

### Mocking Modules

```javascript
// users.js
import axios from 'axios';

export async function getUsers() {
  const response = await axios.get('/users');
  return response.data;
}

// users.test.js
import axios from 'axios';
import { getUsers } from './users';

jest.mock('axios');

test('fetches users', async () => {
  const users = [{ name: 'John' }];
  axios.get.mockResolvedValue({ data: users });

  const result = await getUsers();
  expect(result).toEqual(users);
  expect(axios.get).toHaveBeenCalledWith('/users');
});
```

### Manual Mocks

```javascript
// __mocks__/axios.js
module.exports = {
  get: jest.fn(() => Promise.resolve({ data: {} })),
  post: jest.fn(() => Promise.resolve({ data: {} }))
};

// In test file
jest.mock('axios');
```

### Partial Mocking

```javascript
// Only mock specific functions
jest.mock('./module', () => ({
  ...jest.requireActual('./module'),
  specificFunction: jest.fn()
}));
```

### Spying

```javascript
// Spy on method
const spy = jest.spyOn(object, 'method');

// Original implementation still runs
object.method();

expect(spy).toHaveBeenCalled();

// Restore original
spy.mockRestore();

// Spy and mock implementation
jest.spyOn(object, 'method').mockImplementation(() => 'mocked');
```

## Snapshot Testing

```javascript
// Component rendering
import renderer from 'react-test-renderer';
import Button from './Button';

test('Button renders correctly', () => {
  const tree = renderer.create(<Button>Click me</Button>).toJSON();
  expect(tree).toMatchSnapshot();
});

// Inline snapshots
test('object snapshot', () => {
  const obj = { name: 'John', age: 30 };
  expect(obj).toMatchInlineSnapshot(`
    {
      "age": 30,
      "name": "John",
    }
  `);
});

// Update snapshots
// npm test -- -u
// npm test -- --updateSnapshot
```

## Test Configuration

### Test Filtering

```bash
# Run specific file
npm test -- users.test.js

# Run tests matching pattern
npm test -- --testNamePattern="user"

# Run only changed tests
npm test -- --onlyChanged

# Watch mode
npm test -- --watch

# Run specific test with .only
test.only('this test runs', () => {});

# Skip test with .skip
test.skip('this test skips', () => {});
```

### Test Coverage

```bash
# Generate coverage report
npm test -- --coverage

# Watch with coverage
npm test -- --coverage --watch

# Coverage threshold in jest.config.js
module.exports = {
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  }
};
```

## Testing React Components

```javascript
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import Button from './Button';

describe('Button component', () => {
  test('renders button with text', () => {
    render(<Button>Click me</Button>);
    expect(screen.getByText('Click me')).toBeInTheDocument();
  });

  test('calls onClick when clicked', () => {
    const handleClick = jest.fn();
    render(<Button onClick={handleClick}>Click</Button>);

    fireEvent.click(screen.getByText('Click'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  test('is disabled when disabled prop is true', () => {
    render(<Button disabled>Click</Button>);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  test('renders with correct class', () => {
    render(<Button variant="primary">Click</Button>);
    expect(screen.getByRole('button')).toHaveClass('btn-primary');
  });
});
```

## Timer Mocks

```javascript
// Enable fake timers
jest.useFakeTimers();

test('timer test', () => {
  const callback = jest.fn();
  setTimeout(callback, 1000);

  // Fast-forward time
  jest.advanceTimersByTime(1000);
  expect(callback).toHaveBeenCalled();
});

test('run all timers', () => {
  const callback = jest.fn();
  setTimeout(callback, 1000);

  jest.runAllTimers();
  expect(callback).toHaveBeenCalled();
});

test('run pending timers', () => {
  jest.runOnlyPendingTimers();
});

// Restore real timers
afterAll(() => {
  jest.useRealTimers();
});
```

## Best Practices

### 1. Clear Test Names

```javascript
// Good
test('throws error when user is not found', () => {});

// Bad
test('error test', () => {});
```

### 2. AAA Pattern

```javascript
test('user creation', () => {
  // Arrange
  const userData = { name: 'John', age: 30 };

  // Act
  const user = createUser(userData);

  // Assert
  expect(user.name).toBe('John');
  expect(user.age).toBe(30);
});
```

### 3. One Assertion Per Test (When Practical)

```javascript
// Good - focused tests
test('user has correct name', () => {
  expect(user.name).toBe('John');
});

test('user has correct age', () => {
  expect(user.age).toBe(30);
});

// Acceptable - related assertions
test('user is created with correct properties', () => {
  expect(user.name).toBe('John');
  expect(user.age).toBe(30);
});
```

### 4. Clean Up Mocks

```javascript
describe('tests', () => {
  afterEach(() => {
    jest.clearAllMocks(); // Clear mock call history
    jest.restoreAllMocks(); // Restore original implementations
  });
});
```

### 5. Test Edge Cases

```javascript
describe('divide function', () => {
  test('divides positive numbers', () => {
    expect(divide(10, 2)).toBe(5);
  });

  test('divides negative numbers', () => {
    expect(divide(-10, 2)).toBe(-5);
  });

  test('throws on division by zero', () => {
    expect(() => divide(10, 0)).toThrow();
  });

  test('handles floating point', () => {
    expect(divide(1, 3)).toBeCloseTo(0.333, 2);
  });
});
```

## Common Patterns

### Testing Promises

```javascript
// Chain promises
test('promise chain', () => {
  return fetchUser(1)
    .then(user => fetchPosts(user.id))
    .then(posts => {
      expect(posts.length).toBeGreaterThan(0);
    });
});

// Async/await (recommended)
test('async user and posts', async () => {
  const user = await fetchUser(1);
  const posts = await fetchPosts(user.id);
  expect(posts.length).toBeGreaterThan(0);
});
```

### Testing Errors

```javascript
test('throws on invalid input', () => {
  expect(() => {
    validateEmail('invalid');
  }).toThrow('Invalid email');
});

test('async error handling', async () => {
  await expect(fetchUser(999)).rejects.toThrow('User not found');
});
```

### Testing With Context

```javascript
describe('Calculator', () => {
  let calculator;

  beforeEach(() => {
    calculator = new Calculator();
  });

  describe('add', () => {
    test('adds two numbers', () => {
      expect(calculator.add(2, 3)).toBe(5);
    });
  });

  describe('subtract', () => {
    test('subtracts two numbers', () => {
      expect(calculator.subtract(5, 3)).toBe(2);
    });
  });
});
```

## TypeScript Configuration

```typescript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  transform: {
    '^.+\\.ts$': 'ts-jest'
  }
};

// Example TypeScript test
import { sum } from './math';

describe('sum', () => {
  test('adds numbers', () => {
    const result: number = sum(1, 2);
    expect(result).toBe(3);
  });
});
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `npm test` | Run all tests |
| `npm test -- --watch` | Watch mode |
| `npm test -- --coverage` | Coverage report |
| `npm test -- -u` | Update snapshots |
| `npm test -- --verbose` | Verbose output |
| `npm test -- file.test.js` | Run specific file |
| `test.only()` | Run only this test |
| `test.skip()` | Skip this test |

## Common Matchers

```javascript
expect(value).toBe(expected)              // Strict equality
expect(value).toEqual(expected)           // Deep equality
expect(value).toBeTruthy()                // Truthy
expect(value).toBeNull()                  // Null
expect(value).toBeUndefined()             // Undefined
expect(value).toContain(item)             // Array/string contains
expect(value).toHaveLength(number)        // Length check
expect(fn).toThrow(error)                 // Throws error
expect(promise).resolves.toBe(value)      // Promise resolves
expect(promise).rejects.toThrow()         // Promise rejects
expect(fn).toHaveBeenCalled()             // Mock called
expect(fn).toHaveBeenCalledWith(args)     // Mock called with args
```

## Further Resources

- [Jest Documentation](https://jestjs.io/)
- [Testing Library](https://testing-library.com/)
- [Jest Cheat Sheet](https://github.com/sapegin/jest-cheat-sheet)
- [Common Testing Mistakes](https://kentcdodds.com/blog/common-mistakes-with-react-testing-library)
