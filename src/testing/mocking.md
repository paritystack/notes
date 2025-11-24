# Mocking

Mocking is a testing technique that replaces real dependencies with controlled substitutes to isolate the code under test.

## What is Mocking?

**Mocking** creates fake objects that simulate the behavior of real dependencies, allowing you to:
- Test code in isolation
- Control external dependencies
- Verify interactions between objects
- Speed up test execution
- Test error scenarios

## Test Doubles: Types

### Mock
Verifies that specific methods are called with expected arguments.

```python
from unittest.mock import Mock

# Python
mock = Mock()
service.process(mock)
mock.save.assert_called_once_with(data)
```

```javascript
// JavaScript
const mock = jest.fn();
service.process(mock);
expect(mock).toHaveBeenCalledWith(data);
```

### Stub
Returns predetermined responses to method calls.

```python
# Python
stub = Mock()
stub.get_user.return_value = {"id": 1, "name": "John"}

result = stub.get_user(1)
assert result["name"] == "John"
```

```javascript
// JavaScript
const stub = jest.fn().mockReturnValue({ id: 1, name: "John" });
const result = stub(1);
expect(result.name).toBe("John");
```

### Spy
Records information about how it was called, but calls through to real implementation.

```python
# Python
from unittest.mock import MagicMock

obj = RealObject()
obj.method = MagicMock(wraps=obj.method)

obj.method(arg)  # Real method executes
obj.method.assert_called_once_with(arg)
```

```javascript
// JavaScript
const spy = jest.spyOn(object, 'method');
object.method(arg);  // Real method executes
expect(spy).toHaveBeenCalledWith(arg);
spy.mockRestore();
```

### Fake
Working implementation, but simplified (e.g., in-memory database).

```python
# Python
class FakeDatabase:
    def __init__(self):
        self.data = {}

    def save(self, id, value):
        self.data[id] = value

    def get(self, id):
        return self.data.get(id)

# Use in tests
db = FakeDatabase()
db.save(1, "test")
assert db.get(1) == "test"
```

```javascript
// JavaScript
class FakeDatabase {
  constructor() {
    this.data = new Map();
  }

  save(id, value) {
    this.data.set(id, value);
  }

  get(id) {
    return this.data.get(id);
  }
}

const db = new FakeDatabase();
db.save(1, "test");
expect(db.get(1)).toBe("test");
```

## Python Mocking

### unittest.mock

```python
from unittest.mock import Mock, MagicMock, patch, call

# Basic mock
mock = Mock()
mock.method()
mock.method.assert_called_once()

# Return value
mock.get_user.return_value = {"name": "John"}
assert mock.get_user()["name"] == "John"

# Side effect (multiple returns)
mock.fetch.side_effect = [1, 2, 3]
assert mock.fetch() == 1
assert mock.fetch() == 2
assert mock.fetch() == 3

# Side effect (exception)
mock.validate.side_effect = ValueError("Invalid")
with pytest.raises(ValueError):
    mock.validate()

# Verify calls
mock.method(1, 2, key="value")
mock.method.assert_called_with(1, 2, key="value")
mock.method.assert_called_once_with(1, 2, key="value")

# Check all calls
mock.method(1)
mock.method(2)
mock.method.assert_has_calls([call(1), call(2)])

# Check call count
assert mock.method.call_count == 2
```

### Patching

```python
from unittest.mock import patch

# Patch function
@patch('module.function')
def test_something(mock_function):
    mock_function.return_value = 42
    result = module.function()
    assert result == 42

# Patch as context manager
def test_something():
    with patch('module.function') as mock_function:
        mock_function.return_value = 42
        result = module.function()
        assert result == 42

# Patch class
@patch('module.ClassName')
def test_class(MockClass):
    instance = MockClass.return_value
    instance.method.return_value = "mocked"

    obj = module.ClassName()
    assert obj.method() == "mocked"

# Patch multiple
@patch('module.function2')
@patch('module.function1')
def test_multiple(mock_func1, mock_func2):
    # Note: decorators apply bottom-to-top
    pass

# Patch object method
@patch.object(MyClass, 'method')
def test_method(mock_method):
    mock_method.return_value = "mocked"
    obj = MyClass()
    assert obj.method() == "mocked"

# Patch dictionary
with patch.dict('os.environ', {'API_KEY': 'test'}):
    # os.environ['API_KEY'] is 'test'
    pass
```

### MagicMock

```python
from unittest.mock import MagicMock

# Supports magic methods
mock = MagicMock()

# Context manager
with mock:
    pass
mock.__enter__.assert_called_once()
mock.__exit__.assert_called_once()

# Iterator
mock.__iter__.return_value = iter([1, 2, 3])
assert list(mock) == [1, 2, 3]

# Comparison
mock.__lt__.return_value = True
assert mock < 5

# Container
mock.__getitem__.return_value = "value"
assert mock[0] == "value"
```

### pytest-mock

```python
# pytest-mock plugin provides mocker fixture
def test_with_mocker(mocker):
    # Patch
    mock = mocker.patch('module.function')
    mock.return_value = 42

    # Spy
    spy = mocker.spy(obj, 'method')
    obj.method()
    spy.assert_called_once()

    # Mock open
    m = mocker.mock_open(read_data='content')
    with mocker.patch('builtins.open', m):
        with open('file.txt') as f:
            assert f.read() == 'content'
```

## JavaScript Mocking

### Jest Mocks

```javascript
// Mock function
const mockFn = jest.fn();
mockFn.mockReturnValue(42);
expect(mockFn()).toBe(42);

// Mock implementation
const mockCallback = jest.fn(x => x * 2);
expect(mockCallback(21)).toBe(42);

// Multiple return values
mockFn
  .mockReturnValueOnce(1)
  .mockReturnValueOnce(2)
  .mockReturnValue(3);

expect(mockFn()).toBe(1);
expect(mockFn()).toBe(2);
expect(mockFn()).toBe(3);
expect(mockFn()).toBe(3);

// Async mocks
mockFn.mockResolvedValue('success');
await expect(mockFn()).resolves.toBe('success');

mockFn.mockRejectedValue(new Error('failed'));
await expect(mockFn()).rejects.toThrow('failed');

// Assertions
expect(mockFn).toHaveBeenCalled();
expect(mockFn).toHaveBeenCalledTimes(1);
expect(mockFn).toHaveBeenCalledWith(arg1, arg2);
expect(mockFn).toHaveBeenLastCalledWith(arg);
expect(mockFn).toHaveBeenNthCalledWith(2, arg);

// Access calls
expect(mockFn.mock.calls.length).toBe(2);
expect(mockFn.mock.calls[0][0]).toBe('arg1');
expect(mockFn.mock.results[0].value).toBe(42);
```

### Mocking Modules

```javascript
// Automatic mock
jest.mock('./module');
import { functionA } from './module';
// functionA is automatically mocked

// Manual mock implementation
jest.mock('./module', () => ({
  functionA: jest.fn(() => 'mocked'),
  functionB: jest.fn()
}));

// Partial mock
jest.mock('./module', () => ({
  ...jest.requireActual('./module'),
  functionA: jest.fn()  // Only functionA is mocked
}));

// Mock default export
jest.mock('./module', () => ({
  __esModule: true,
  default: jest.fn(() => 'mocked')
}));
```

### Manual Mocks

```javascript
// __mocks__/axios.js
export default {
  get: jest.fn(() => Promise.resolve({ data: {} })),
  post: jest.fn(() => Promise.resolve({ data: {} }))
};

// test.js
jest.mock('axios');
import axios from 'axios';

test('fetches data', async () => {
  axios.get.mockResolvedValue({ data: { name: 'John' } });
  const result = await fetchUser(1);
  expect(result.name).toBe('John');
});
```

### Spying

```javascript
const obj = {
  method: () => 'original'
};

// Spy on method
const spy = jest.spyOn(obj, 'method');
expect(obj.method()).toBe('original');  // Original runs
expect(spy).toHaveBeenCalled();

// Spy with mock implementation
jest.spyOn(obj, 'method').mockImplementation(() => 'mocked');
expect(obj.method()).toBe('mocked');

// Restore original
spy.mockRestore();
expect(obj.method()).toBe('original');
```

## Mocking Common Dependencies

### HTTP Requests

**Python:**
```python
import requests
from unittest.mock import patch

@patch('requests.get')
def test_api_call(mock_get):
    mock_get.return_value.json.return_value = {'id': 1}
    mock_get.return_value.status_code = 200

    response = requests.get('https://api.example.com/users/1')
    assert response.json()['id'] == 1

    mock_get.assert_called_once_with('https://api.example.com/users/1')
```

**JavaScript:**
```javascript
import axios from 'axios';
jest.mock('axios');

test('API call', async () => {
  axios.get.mockResolvedValue({
    data: { id: 1 },
    status: 200
  });

  const response = await axios.get('https://api.example.com/users/1');
  expect(response.data.id).toBe(1);
  expect(axios.get).toHaveBeenCalledWith('https://api.example.com/users/1');
});
```

### Database

**Python:**
```python
from unittest.mock import Mock, patch

@patch('app.database.get_connection')
def test_user_fetch(mock_get_conn):
    # Mock connection and cursor
    mock_cursor = Mock()
    mock_cursor.fetchone.return_value = (1, 'John', 'john@example.com')

    mock_conn = Mock()
    mock_conn.cursor.return_value = mock_cursor

    mock_get_conn.return_value = mock_conn

    # Test function
    user = fetch_user(1)

    assert user['name'] == 'John'
    mock_cursor.execute.assert_called_once()
```

**JavaScript:**
```javascript
const db = {
  query: jest.fn()
};

test('database query', async () => {
  db.query.mockResolvedValue([
    { id: 1, name: 'John' }
  ]);

  const users = await getUsers();

  expect(users[0].name).toBe('John');
  expect(db.query).toHaveBeenCalledWith('SELECT * FROM users');
});
```

### File System

**Python:**
```python
from unittest.mock import mock_open, patch

def test_read_file():
    mock_data = "file content"

    with patch('builtins.open', mock_open(read_data=mock_data)):
        with open('test.txt') as f:
            content = f.read()

    assert content == "file content"

def test_write_file():
    m = mock_open()

    with patch('builtins.open', m):
        with open('test.txt', 'w') as f:
            f.write('content')

    m.assert_called_once_with('test.txt', 'w')
    m().write.assert_called_once_with('content')
```

**JavaScript:**
```javascript
import fs from 'fs';
jest.mock('fs');

test('read file', () => {
  fs.readFileSync.mockReturnValue('file content');

  const content = fs.readFileSync('test.txt', 'utf8');

  expect(content).toBe('file content');
  expect(fs.readFileSync).toHaveBeenCalledWith('test.txt', 'utf8');
});

test('write file', () => {
  fs.writeFileSync.mockImplementation(() => {});

  fs.writeFileSync('test.txt', 'content');

  expect(fs.writeFileSync).toHaveBeenCalledWith('test.txt', 'content');
});
```

### Time and Dates

**Python:**
```python
from datetime import datetime
from unittest.mock import patch

@patch('module.datetime')
def test_with_fixed_time(mock_datetime):
    # Fix time to specific date
    mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)

    result = get_current_date()
    assert result == datetime(2024, 1, 1, 12, 0, 0)
```

**JavaScript:**
```javascript
// Jest fake timers
jest.useFakeTimers();
jest.setSystemTime(new Date('2024-01-01'));

test('with fixed date', () => {
  const now = new Date();
  expect(now.getFullYear()).toBe(2024);
  expect(now.getMonth()).toBe(0);
});

jest.useRealTimers();

// Mock Date constructor
const mockDate = new Date('2024-01-01');
global.Date = jest.fn(() => mockDate);

test('date test', () => {
  const date = new Date();
  expect(date.getFullYear()).toBe(2024);
});
```

### Environment Variables

**Python:**
```python
from unittest.mock import patch

@patch.dict('os.environ', {'API_KEY': 'test_key', 'DEBUG': 'true'})
def test_with_env():
    import os
    assert os.environ['API_KEY'] == 'test_key'
    assert os.environ['DEBUG'] == 'true'
```

**JavaScript:**
```javascript
const originalEnv = process.env;

beforeEach(() => {
  jest.resetModules();
  process.env = { ...originalEnv };
});

afterEach(() => {
  process.env = originalEnv;
});

test('with env vars', () => {
  process.env.API_KEY = 'test_key';
  process.env.DEBUG = 'true';

  expect(process.env.API_KEY).toBe('test_key');
  expect(process.env.DEBUG).toBe('true');
});
```

### Random Values

**Python:**
```python
from unittest.mock import patch
import random

@patch('random.randint')
def test_random(mock_randint):
    mock_randint.return_value = 7

    result = random.randint(1, 10)
    assert result == 7
```

**JavaScript:**
```javascript
const spy = jest.spyOn(Math, 'random');
spy.mockReturnValue(0.5);

test('random value', () => {
  expect(Math.random()).toBe(0.5);
});

spy.mockRestore();
```

## Dependency Injection

Making code testable through dependency injection:

**Before (Hard to test):**
```python
class UserService:
    def get_user(self, user_id):
        # Hard-coded dependency
        db = Database()
        return db.query(f"SELECT * FROM users WHERE id={user_id}")
```

**After (Easy to test):**
```python
class UserService:
    def __init__(self, database):
        self.database = database

    def get_user(self, user_id):
        return self.database.query(f"SELECT * FROM users WHERE id={user_id}")

# Test
def test_get_user():
    mock_db = Mock()
    mock_db.query.return_value = {"id": 1, "name": "John"}

    service = UserService(mock_db)
    user = service.get_user(1)

    assert user["name"] == "John"
```

**JavaScript:**
```javascript
// Before
class UserService {
  getUser(id) {
    const db = new Database();  // Hard-coded
    return db.query(`SELECT * FROM users WHERE id=${id}`);
  }
}

// After
class UserService {
  constructor(database) {
    this.database = database;
  }

  getUser(id) {
    return this.database.query(`SELECT * FROM users WHERE id=${id}`);
  }
}

// Test
test('getUser', () => {
  const mockDb = {
    query: jest.fn().mockReturnValue({ id: 1, name: 'John' })
  };

  const service = new UserService(mockDb);
  const user = service.getUser(1);

  expect(user.name).toBe('John');
});
```

## Best Practices

### 1. Mock at the Right Level

```python
# Bad - Mocking too low level
@patch('json.dumps')
def test_api(mock_dumps):
    pass

# Good - Mock at boundary
@patch('requests.post')
def test_api(mock_post):
    pass
```

### 2. Don't Mock What You Don't Own

```python
# Bad - Mocking internal Python
@patch('builtins.len')
def test_something(mock_len):
    pass

# Good - Mock your own code
@patch('myapp.custom_length_function')
def test_something(mock_length):
    pass
```

### 3. Verify Behavior, Not Implementation

```python
# Bad - Testing implementation details
def test_process():
    mock = Mock()
    service.process(mock)

    # Too specific
    assert mock.method1.call_count == 1
    assert mock.method2.call_count == 2

# Good - Testing behavior
def test_process():
    mock = Mock()
    service.process(mock)

    # Verify outcome
    mock.save.assert_called_once()
```

### 4. Use Realistic Test Data

```python
# Bad - Unrealistic data
mock.get_user.return_value = {"id": 1}

# Good - Complete realistic data
mock.get_user.return_value = {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "created_at": "2024-01-01T00:00:00Z"
}
```

### 5. Clean Up Mocks

```python
# Python - Use fixtures
@pytest.fixture
def mock_api():
    with patch('module.api') as mock:
        yield mock
    # Automatically cleaned up

# JavaScript - afterEach
afterEach(() => {
  jest.clearAllMocks();
  jest.restoreAllMocks();
});
```

### 6. Mock Only What's Necessary

```python
# Bad - Over-mocking
@patch('module.function1')
@patch('module.function2')
@patch('module.function3')
def test_something(m3, m2, m1):
    # Only using m1
    pass

# Good - Mock what you need
@patch('module.function1')
def test_something(mock_func):
    pass
```

## Anti-Patterns

### 1. Mocking Everything

```python
# Bad
@patch('module.A')
@patch('module.B')
@patch('module.C')
@patch('module.D')
def test_everything_mocked(d, c, b, a):
    # Not testing real code interactions
    pass
```

### 2. Not Verifying Mock Calls

```javascript
// Bad
const mock = jest.fn();
service.process(mock);
// No verification

// Good
const mock = jest.fn();
service.process(mock);
expect(mock).toHaveBeenCalled();
```

### 3. Brittle Mocks

```python
# Bad - Too specific
mock.method.assert_called_once_with(
    arg1="exact value",
    arg2=123,
    arg3=True,
    arg4=[1, 2, 3]
)

# Good - Test what matters
from unittest.mock import ANY

mock.method.assert_called_once_with(
    arg1=ANY,  # Don't care about exact value
    arg2=123   # Care about this
)
```

## Quick Reference

### Python unittest.mock

```python
from unittest.mock import Mock, patch, MagicMock, call, ANY

# Create mocks
mock = Mock()
magic = MagicMock()

# Return values
mock.method.return_value = 'value'
mock.method.side_effect = [1, 2, 3]
mock.method.side_effect = Exception()

# Assertions
mock.assert_called()
mock.assert_called_once()
mock.assert_called_with(args)
mock.assert_called_once_with(args)
mock.assert_has_calls([call(1), call(2)])
assert mock.call_count == 2

# Patching
@patch('module.Class')
with patch('module.function') as mock:
```

### Jest

```javascript
// Create mocks
const mock = jest.fn();

// Return values
mock.mockReturnValue('value');
mock.mockResolvedValue('async value');
mock.mockRejectedValue(new Error());

// Assertions
expect(mock).toHaveBeenCalled();
expect(mock).toHaveBeenCalledTimes(1);
expect(mock).toHaveBeenCalledWith(args);
expect(mock).toHaveBeenLastCalledWith(args);

// Module mocking
jest.mock('./module');
jest.spyOn(obj, 'method');
```

## Further Resources

- [Python Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Jest Mock Documentation](https://jestjs.io/docs/mock-functions)
- [pytest-mock](https://pytest-mock.readthedocs.io/)
- [Testing with Mocks](https://martinfowler.com/articles/mocksArentStubs.html)
