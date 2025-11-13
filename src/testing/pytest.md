# pytest

pytest is a mature, feature-rich testing framework for Python that makes it easy to write simple tests, yet scales to support complex functional testing.

## Installation

```bash
pip install pytest
pip install pytest pytest-cov pytest-mock pytest-xdist

# Verify
pytest --version
```

## Basic Usage

```bash
# Run all tests
pytest

# Run specific file
pytest test_example.py

# Run specific test
pytest test_example.py::test_function

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=myapp tests/

# Parallel execution
pytest -n 4
```

## Writing Tests

```python
# test_example.py

# Simple test
def test_addition():
    assert 1 + 1 == 2

# Test with setup
def test_list():
    my_list = [1, 2, 3]
    assert len(my_list) == 3
    assert 2 in my_list

# Test exceptions
import pytest

def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0

# Parametrized test
@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 3),
    (3, 4),
])
def test_increment(input, expected):
    assert input + 1 == expected
```

## Fixtures

```python
import pytest

# Basic fixture
@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5]

def test_sum(sample_data):
    assert sum(sample_data) == 15

# Fixture with setup/teardown
@pytest.fixture
def database_connection():
    # Setup
    conn = create_connection()
    yield conn
    # Teardown
    conn.close()

# Scope: function (default), class, module, package, session
@pytest.fixture(scope="module")
def expensive_resource():
    return load_expensive_data()

# Autouse fixture
@pytest.fixture(autouse=True)
def setup_test():
    print("Setting up test")
    yield
    print("Tearing down test")
```

## Markers

```python
import pytest

# Skip test
@pytest.mark.skip(reason="Not implemented yet")
def test_feature():
    pass

# Skip conditionally
@pytest.mark.skipif(sys.version_info < (3, 8), reason="Requires Python 3.8+")
def test_modern_feature():
    pass

# Expected to fail
@pytest.mark.xfail
def test_known_bug():
    assert False

# Custom marker
@pytest.mark.slow
def test_slow_operation():
    pass

# Run specific markers
# pytest -m slow
# pytest -m "not slow"
```

## Mocking

```python
from unittest.mock import Mock, patch, MagicMock

def test_with_mock():
    mock_obj = Mock()
    mock_obj.method.return_value = 42
    assert mock_obj.method() == 42

# Patch function
def test_with_patch():
    with patch('module.function') as mock_func:
        mock_func.return_value = 'mocked'
        result = module.function()
        assert result == 'mocked'

# pytest-mock plugin
def test_with_mocker(mocker):
    mock = mocker.patch('module.function')
    mock.return_value = 'mocked'
    assert module.function() == 'mocked'
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `pytest` | Run all tests |
| `pytest -v` | Verbose output |
| `pytest -k pattern` | Run tests matching pattern |
| `pytest -m marker` | Run tests with marker |
| `pytest --cov` | Coverage report |
| `pytest -x` | Stop on first failure |
| `pytest --pdb` | Drop into debugger on failure |

pytest is the de facto standard for Python testing with its simple syntax, powerful features, and extensive plugin ecosystem.
