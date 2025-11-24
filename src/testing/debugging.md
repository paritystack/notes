# Debugging Tests

Debugging test failures is a critical skill for maintaining a healthy test suite. This guide covers strategies and tools for finding and fixing test issues.

## Common Test Failure Patterns

### 1. Assertion Failures

**Symptom**: Expected value doesn't match actual value.

```python
# Python
def test_calculation():
    result = calculate_total(100, 0.1)
    assert result == 110  # FAIL: AssertionError: assert 90 == 110

# Debug: Print values
def test_calculation():
    result = calculate_total(100, 0.1)
    print(f"Result: {result}")  # Shows actual value
    assert result == 110
```

```javascript
// JavaScript
test('calculation', () => {
  const result = calculateTotal(100, 0.1);
  expect(result).toBe(110);  // FAIL: Expected: 110, Received: 90
});

// Debug: Console log
test('calculation', () => {
  const result = calculateTotal(100, 0.1);
  console.log('Result:', result);
  expect(result).toBe(110);
});
```

### 2. Type Errors

```python
# Python
def test_user():
    user = get_user(1)
    assert user['name'] == 'John'  # TypeError: 'NoneType' object is not subscriptable

# Debug: Check type
def test_user():
    user = get_user(1)
    print(f"User type: {type(user)}, value: {user}")
    assert user is not None, "User should not be None"
    assert user['name'] == 'John'
```

```javascript
// JavaScript
test('user', () => {
  const user = getUser(1);
  expect(user.name).toBe('John');  // TypeError: Cannot read property 'name' of undefined
});

// Debug: Check existence
test('user', () => {
  const user = getUser(1);
  console.log('User:', user);
  expect(user).toBeDefined();
  expect(user.name).toBe('John');
});
```

### 3. Async Issues

```python
# Python
def test_async_fetch():
    result = fetch_data()  # Returns coroutine, not value
    assert result == 'data'  # Fails

# Fix: Use async/await
import pytest

@pytest.mark.asyncio
async def test_async_fetch():
    result = await fetch_data()
    assert result == 'data'
```

```javascript
// JavaScript
test('async fetch', () => {
  const result = fetchData();  // Returns Promise
  expect(result).toBe('data');  // Fails
});

// Fix: Use async/await
test('async fetch', async () => {
  const result = await fetchData();
  expect(result).toBe('data');
});
```

### 4. Flaky Tests

Tests that pass/fail inconsistently.

**Common causes:**
- Race conditions
- Timing issues
- Shared state
- Random data
- External dependencies

```python
# Flaky test
def test_fetch():
    result = fetch_from_api()  # Sometimes fails due to network
    assert result['status'] == 'ok'

# Fix: Mock external dependency
from unittest.mock import patch

@patch('module.fetch_from_api')
def test_fetch(mock_fetch):
    mock_fetch.return_value = {'status': 'ok'}
    result = fetch_from_api()
    assert result['status'] == 'ok'
```

### 5. Order Dependencies

```python
# Bad: Tests depend on order
def test_create_user():
    create_user('john')

def test_user_exists():
    assert user_exists('john')  # Fails if run alone

# Good: Independent tests
def test_create_user():
    create_user('john')
    delete_user('john')  # Cleanup

def test_user_exists():
    create_user('test')  # Own setup
    assert user_exists('test')
    delete_user('test')
```

## Debugging Tools

### Python Debugger (pdb)

**Basic usage:**
```python
import pdb

def test_calculation():
    result = calculate_total(100, 0.1)
    pdb.set_trace()  # Execution stops here
    assert result == 110
```

**Interactive commands:**
```
(Pdb) n          # Next line
(Pdb) s          # Step into function
(Pdb) c          # Continue execution
(Pdb) l          # List code
(Pdb) p result   # Print variable
(Pdb) pp obj     # Pretty print
(Pdb) w          # Where am I (stack trace)
(Pdb) q          # Quit debugger
```

**pytest with pdb:**
```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of each test
pytest --trace

# Use breakpoint()
def test_something():
    breakpoint()  # Python 3.7+
    assert True
```

### Enhanced Python Debugger (ipdb)

```bash
pip install ipdb
```

```python
import ipdb

def test_calculation():
    result = calculate_total(100, 0.1)
    ipdb.set_trace()
    assert result == 110
```

**Features:**
- Syntax highlighting
- Tab completion
- Better formatting

### Node.js Debugger

**Basic debugging:**
```javascript
// Add debugger statement
test('calculation', () => {
  const result = calculateTotal(100, 0.1);
  debugger;  // Execution stops here
  expect(result).toBe(110);
});
```

```bash
# Run with debugger
node --inspect-brk node_modules/.bin/jest --runInBand

# Chrome DevTools
# Open chrome://inspect
```

**Jest debugging:**
```bash
# Debug specific test
node --inspect-brk node_modules/.bin/jest --runInBand test/specific.test.js

# VS Code: Add to launch.json
{
  "type": "node",
  "request": "launch",
  "name": "Jest Debug",
  "program": "${workspaceFolder}/node_modules/.bin/jest",
  "args": ["--runInBand", "--no-cache"],
  "console": "integratedTerminal"
}
```

## IDE Debugging

### VS Code

**Python configuration (.vscode/launch.json):**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "-v",
        "${file}"
      ],
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

**JavaScript configuration:**
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Jest: current file",
      "program": "${workspaceFolder}/node_modules/.bin/jest",
      "args": [
        "${fileBasenameNoExtension}",
        "--config",
        "jest.config.js"
      ],
      "console": "integratedTerminal",
      "internalConsoleOptions": "neverOpen"
    }
  ]
}
```

**Usage:**
1. Set breakpoints (click left of line number)
2. Run debugger (F5)
3. Step through code
4. Inspect variables

### PyCharm

**Run test with debugger:**
1. Right-click test function
2. Select "Debug 'test_name'"
3. Use debugger controls

**Breakpoints:**
- Click left gutter to set breakpoint
- Right-click breakpoint for conditions
- View variables in debugger panel

### IntelliJ IDEA / WebStorm

**JavaScript debugging:**
1. Right-click test file
2. Select "Debug 'test-file.test.js'"
3. Set breakpoints
4. Use debugger controls

## Print Debugging

Sometimes simple print statements are most effective.

### Python

```python
def test_complex_logic():
    data = fetch_data()
    print(f"\n=== Debug Info ===")
    print(f"Data type: {type(data)}")
    print(f"Data value: {data}")
    print(f"Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    print(f"=================\n")

    result = process(data)
    print(f"Result: {result}")

    assert result == expected
```

**pytest output:**
```bash
# Show print statements
pytest -s

# Verbose output
pytest -v

# Show local variables on failure
pytest -l

# Combine flags
pytest -svl
```

### JavaScript

```javascript
test('complex logic', () => {
  const data = fetchData();
  console.log('=== Debug Info ===');
  console.log('Data type:', typeof data);
  console.log('Data value:', data);
  console.log('Data:', JSON.stringify(data, null, 2));
  console.log('==================');

  const result = process(data);
  console.log('Result:', result);

  expect(result).toBe(expected);
});
```

**Jest output:**
```bash
# Show console output
npm test -- --verbose

# No output buffering
npm test -- --no-coverage
```

## Logging in Tests

### Python: logging module

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_with_logging():
    logger.debug("Starting test")
    result = complex_operation()
    logger.info(f"Operation result: {result}")
    logger.warning("This is unusual")
    assert result > 0
```

**pytest-logging:**
```python
def test_logs(caplog):
    with caplog.at_level(logging.INFO):
        function_that_logs()

    assert "Expected message" in caplog.text
    assert caplog.records[0].levelname == "INFO"
```

### JavaScript: debug module

```bash
npm install debug
```

```javascript
const debug = require('debug')('test:user');

test('with debugging', () => {
  debug('Starting test');
  const result = complexOperation();
  debug('Result: %O', result);
  expect(result).toBeDefined();
});
```

```bash
# Run with debug output
DEBUG=test:* npm test
```

## Isolating Test Failures

### Run Single Test

```bash
# Python
pytest tests/test_file.py::test_function
pytest tests/test_file.py::TestClass::test_method
pytest -k "test_name"  # Match test name

# JavaScript (Jest)
npm test -- test-file.test.js
npm test -- --testNamePattern="test name"
npm test -- -t "test name"
```

### Skip Other Tests

```python
# Python
import pytest

@pytest.mark.skip(reason="Not ready")
def test_not_ready():
    pass

def test_focus():
    # Only this test runs
    pass
```

```javascript
// JavaScript
test.skip('not ready', () => {
  // This test skips
});

test.only('focus', () => {
  // Only this test runs
});
```

### Run Failed Tests Only

```bash
# Python
pytest --lf  # Last failed
pytest --ff  # Failed first

# JavaScript (Jest)
npm test -- --onlyFailures
```

## Debugging Techniques

### 1. Binary Search

Narrow down the problem by commenting out code.

```python
def test_complex():
    # result = step1()
    # result = step2(result)
    # result = step3(result)
    result = step4()  # Problem is in step4 or later
    assert result == expected
```

### 2. Simplify Test

Create minimal reproduction.

```python
# Original failing test
def test_complex_scenario():
    setup_database()
    create_users(100)
    result = complex_operation()
    assert result == expected

# Simplified
def test_minimal():
    result = complex_operation()  # Still fails?
    assert result == expected
```

### 3. Compare Working vs Failing

```python
# Working test
def test_works():
    result = function(valid_input)
    assert result == expected

# Failing test
def test_fails():
    result = function(invalid_input)
    print(f"Difference: valid={valid_input}, invalid={invalid_input}")
    assert result == expected
```

### 4. Check Assumptions

```python
def test_with_checks():
    data = fetch_data()

    # Verify assumptions
    assert data is not None, "Data should not be None"
    assert isinstance(data, list), f"Data should be list, got {type(data)}"
    assert len(data) > 0, "Data should not be empty"

    result = process(data)
    assert result == expected
```

### 5. Add Intermediate Assertions

```python
def test_multi_step():
    step1_result = step1()
    assert step1_result is not None  # Checkpoint

    step2_result = step2(step1_result)
    assert step2_result > 0  # Checkpoint

    final = step3(step2_result)
    assert final == expected
```

## Common Debugging Scenarios

### Mock Not Working

```python
# Problem: Mock not being used
from unittest.mock import patch

@patch('mymodule.function')  # Wrong: patches mymodule.function
def test_something(mock_func):
    from othermodule import uses_function
    uses_function()  # Doesn't use the mock!

# Solution: Patch where it's used
@patch('othermodule.function')  # Patch where it's imported
def test_something(mock_func):
    from othermodule import uses_function
    uses_function()  # Now uses the mock
```

### Fixture Not Running

```python
# Problem: Fixture not applied
@pytest.fixture
def setup_db():
    initialize_database()
    yield
    cleanup_database()

def test_user():  # Fixture not used!
    user = create_user()

# Solution: Use fixture
def test_user(setup_db):  # Now fixture runs
    user = create_user()
```

### Async Test Not Awaited

```python
# Problem
import pytest

@pytest.mark.asyncio
def test_async():  # Missing async keyword
    result = await fetch_data()  # SyntaxError

# Solution
@pytest.mark.asyncio
async def test_async():
    result = await fetch_data()
```

### State Pollution

```python
# Problem: Tests affecting each other
shared_list = []

def test_a():
    shared_list.append(1)
    assert len(shared_list) == 1

def test_b():
    shared_list.append(2)
    assert len(shared_list) == 1  # Fails if test_a ran first

# Solution: Clean state
@pytest.fixture(autouse=True)
def clean_state():
    shared_list.clear()
    yield
    shared_list.clear()
```

## Debugging Flaky Tests

### 1. Run Multiple Times

```bash
# Python
pytest --count=100 test_flaky.py

# JavaScript
npm test -- --testPathPattern=flaky --maxWorkers=1 --testTimeout=10000
```

### 2. Add Delays to Expose Timing Issues

```python
import time

def test_timing():
    trigger_async_operation()
    time.sleep(0.1)  # Small delay
    result = check_result()
    assert result == expected
```

### 3. Isolate from Other Tests

```bash
# Run alone
pytest test_flaky.py

# Run in specific order
pytest test_a.py test_flaky.py test_b.py
```

### 4. Check for Race Conditions

```python
# Add synchronization
import threading

lock = threading.Lock()

def test_concurrent():
    with lock:
        # Critical section
        result = shared_operation()
    assert result == expected
```

## Test Output Analysis

### Python pytest Output

```
================================ test session starts =================================
platform linux -- Python 3.11.0, pytest-7.4.0
collected 5 items

tests/test_user.py .F...                                                     [100%]

====================================== FAILURES ======================================
_________________________________ test_user_create __________________________________

    def test_user_create():
>       assert user.age == 30
E       AssertionError: assert 25 == 30
E        +  where 25 = User(name='John', age=25).age

tests/test_user.py:15: AssertionError
================================ short test summary =================================
FAILED tests/test_user.py::test_user_create - AssertionError: assert 25 == 30
============================== 1 failed, 4 passed in 0.05s ===========================
```

**Key information:**
- `.` = passed, `F` = failed
- Line number of failure
- Expected vs actual values
- Object inspection

### Jest Output

```
 FAIL  tests/user.test.js
  User
    ✓ should create user (5 ms)
    ✕ should validate age (3 ms)
    ✓ should update user (2 ms)

  ● User › should validate age

    expect(received).toBe(expected) // Object.is equality

    Expected: 30
    Received: 25

      12 |   test('should validate age', () => {
      13 |     const user = new User('John', 25);
    > 14 |     expect(user.age).toBe(30);
         |                      ^
      15 |   });

      at Object.<anonymous> (tests/user.test.js:14:22)

Test Suites: 1 failed, 1 total
Tests:       1 failed, 2 passed, 3 total
```

## Quick Reference

### Python Debugging Commands

```bash
pytest -v                # Verbose
pytest -s                # Show print statements
pytest -l                # Show local variables
pytest --pdb             # Drop into debugger on failure
pytest --trace           # Start debugger immediately
pytest --lf              # Run last failed
pytest --ff              # Run failed first
pytest -k "test_name"    # Run specific tests
pytest -x                # Stop on first failure
pytest --maxfail=2       # Stop after 2 failures
```

### JavaScript Debugging Commands

```bash
npm test -- --verbose            # Verbose output
npm test -- -t "test name"       # Run specific test
npm test -- --onlyFailures       # Run failed tests
npm test -- --detectOpenHandles  # Find async issues
npm test -- --runInBand          # Run serially
node --inspect-brk jest          # Debug with DevTools
```

### Debugger Cheatsheet

**pdb/ipdb:**
- `n` - Next line
- `s` - Step into
- `c` - Continue
- `l` - List code
- `p var` - Print variable
- `pp var` - Pretty print
- `w` - Show stack
- `q` - Quit

**Chrome DevTools:**
- F8 - Continue
- F10 - Step over
- F11 - Step into
- Shift+F11 - Step out
- Hover variables to inspect

## Further Resources

- [pytest Documentation](https://docs.pytest.org/en/stable/)
- [pdb Documentation](https://docs.python.org/3/library/pdb.html)
- [Jest Debugging](https://jestjs.io/docs/troubleshooting)
- [Chrome DevTools](https://developer.chrome.com/docs/devtools/)
- [VS Code Debugging](https://code.visualstudio.com/docs/editor/debugging)
