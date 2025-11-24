# Test Coverage

Test coverage measures how much of your code is executed during testing. It helps identify untested code but is not a guarantee of code quality.

## What is Code Coverage?

**Code coverage** is a metric that measures the percentage of your code that is executed when tests run. It helps identify:
- Untested code paths
- Dead code
- Areas needing more tests
- Overall test suite health

## Types of Coverage

### Line Coverage
Percentage of code lines executed during tests.

```python
def calculate_discount(price, is_member):
    if is_member:                    # Line executed
        discount = price * 0.1       # Line executed
    else:
        discount = 0                 # Line NOT executed
    return price - discount          # Line executed

# Test only covers member path: 75% line coverage (3/4 lines)
def test_member_discount():
    assert calculate_discount(100, True) == 90
```

### Branch Coverage
Percentage of decision branches executed.

```python
def get_grade(score):
    if score >= 90:         # Branch 1: True
        return 'A'
    elif score >= 80:       # Branch 2: Not tested
        return 'B'
    else:                   # Branch 3: Not tested
        return 'C'

# Only tests one branch: ~33% branch coverage
def test_grade_a():
    assert get_grade(95) == 'A'
```

### Statement Coverage
Percentage of statements executed (similar to line coverage).

### Function Coverage
Percentage of functions/methods called.

```python
class Calculator:
    def add(self, a, b):        # Called
        return a + b

    def subtract(self, a, b):   # Called
        return a - b

    def multiply(self, a, b):   # NOT called
        return a * b

# Function coverage: 66% (2/3 functions tested)
def test_calculator():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.subtract(5, 3) == 2
```

### Path Coverage
Percentage of possible execution paths tested (most comprehensive).

```python
def process(a, b):
    if a > 0:              # Path 1: True
        x = a * 2
    else:                  # Path 2: False
        x = a

    if b > 0:              # Path 3: True
        y = b * 2
    else:                  # Path 4: False
        y = b

    return x + y

# Possible paths: 4 (2 × 2)
# Need tests for: (True, True), (True, False), (False, True), (False, False)
```

## Coverage Tools

### Python: pytest-cov

**Installation:**
```bash
pip install pytest-cov
```

**Basic usage:**
```bash
# Run tests with coverage
pytest --cov=myapp tests/

# Specify coverage report format
pytest --cov=myapp --cov-report=html tests/
pytest --cov=myapp --cov-report=xml tests/
pytest --cov=myapp --cov-report=term tests/

# Coverage with missing lines
pytest --cov=myapp --cov-report=term-missing tests/

# Coverage for specific modules
pytest --cov=myapp.module tests/
```

**Configuration (.coveragerc):**
```ini
[run]
source = myapp
omit =
    */tests/*
    */venv/*
    */__pycache__/*
    */migrations/*

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

### Python: coverage.py

```bash
# Install
pip install coverage

# Run tests
coverage run -m pytest

# Generate report
coverage report
coverage report -m  # Show missing lines

# HTML report
coverage html
# Open htmlcov/index.html

# Erase previous data
coverage erase
```

### JavaScript: Jest

**Built-in coverage:**
```bash
# Run with coverage
npm test -- --coverage

# Coverage for specific files
npm test -- --coverage --collectCoverageFrom="src/**/*.js"

# Watch mode with coverage
npm test -- --coverage --watch
```

**Configuration (package.json):**
```json
{
  "jest": {
    "coverageDirectory": "coverage",
    "collectCoverageFrom": [
      "src/**/*.{js,jsx,ts,tsx}",
      "!src/**/*.test.{js,jsx,ts,tsx}",
      "!src/index.js"
    ],
    "coverageThreshold": {
      "global": {
        "branches": 80,
        "functions": 80,
        "lines": 80,
        "statements": 80
      }
    }
  }
}
```

### JavaScript: NYC (Istanbul)

```bash
# Install
npm install --save-dev nyc

# Run tests with coverage
nyc npm test

# Specify reporters
nyc --reporter=html --reporter=text npm test

# Configuration
nyc --all --include='src/**/*.js' npm test
```

**Configuration (.nycrc):**
```json
{
  "all": true,
  "include": ["src/**/*.js"],
  "exclude": [
    "**/*.test.js",
    "**/__tests__/**",
    "**/node_modules/**"
  ],
  "reporter": ["html", "text", "lcov"],
  "check-coverage": true,
  "branches": 80,
  "lines": 80,
  "functions": 80,
  "statements": 80
}
```

## Reading Coverage Reports

### Terminal Report

```
Name                      Stmts   Miss  Cover   Missing
-------------------------------------------------------
myapp/__init__.py             2      0   100%
myapp/models.py              45      5    89%   23-27
myapp/views.py               67     12    82%   45, 67-78
myapp/utils.py               23      0   100%
-------------------------------------------------------
TOTAL                       137     17    88%
```

**Columns:**
- **Stmts**: Total statements
- **Miss**: Statements not executed
- **Cover**: Coverage percentage
- **Missing**: Line numbers not covered

### HTML Report

```bash
# Generate HTML report
pytest --cov=myapp --cov-report=html

# Open in browser
open htmlcov/index.html
```

Features:
- File-by-file breakdown
- Highlighted covered/uncovered lines
- Branch coverage visualization
- Sortable columns
- Search functionality

### Coverage Badges

**For GitHub README:**
```markdown
![Coverage](https://img.shields.io/badge/coverage-88%25-green)
```

**Using Codecov:**
```yaml
# .github/workflows/test.yml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Setting Coverage Thresholds

### pytest-cov

```bash
# Fail if below threshold
pytest --cov=myapp --cov-fail-under=80
```

**In pytest.ini:**
```ini
[tool:pytest]
addopts = --cov=myapp --cov-fail-under=80
```

### Jest

```json
{
  "jest": {
    "coverageThreshold": {
      "global": {
        "branches": 80,
        "functions": 80,
        "lines": 80,
        "statements": 80
      },
      "src/critical.js": {
        "branches": 100,
        "functions": 100,
        "lines": 100,
        "statements": 100
      }
    }
  }
}
```

### NYC

```json
{
  "nyc": {
    "check-coverage": true,
    "per-file": true,
    "branches": 80,
    "functions": 80,
    "lines": 80,
    "statements": 80
  }
}
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests with coverage
        run: pytest --cov=myapp --cov-report=xml --cov-report=term

      - name: Check coverage threshold
        run: pytest --cov=myapp --cov-fail-under=80

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true
```

### GitLab CI

```yaml
test:
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest --cov=myapp --cov-report=xml --cov-report=term
    - pytest --cov=myapp --cov-fail-under=80
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Run tests with coverage
pytest --cov=myapp --cov-fail-under=80

if [ $? -ne 0 ]; then
    echo "Coverage below threshold. Commit rejected."
    exit 1
fi
```

## Coverage Best Practices

### 1. Aim for Meaningful Coverage, Not 100%

```python
# Don't waste time testing trivial code
class User:
    def __init__(self, name):
        self.name = name  # No need to test this

    def get_name(self):
        return self.name  # Or this

# Focus on business logic
def calculate_tax(income, deductions):
    # This needs thorough testing
    taxable_income = income - deductions
    if taxable_income <= 0:
        return 0
    elif taxable_income <= 50000:
        return taxable_income * 0.1
    else:
        return 5000 + (taxable_income - 50000) * 0.2
```

### 2. Use Coverage to Find Gaps, Not As a Goal

```python
# Bad: Writing tests just for coverage
def test_unused_function():
    # This function is never called in production
    result = unused_function()
    assert result is not None  # Pointless test

# Good: Use coverage to identify untested critical paths
def test_payment_processing():
    # Coverage shows this path isn't tested
    result = process_payment(amount=-100)
    assert result.error == "Invalid amount"
```

### 3. Focus on Branch Coverage

```python
def apply_discount(price, code):
    if code == "SAVE10":
        return price * 0.9
    elif code == "SAVE20":
        return price * 0.8
    else:
        return price

# Need tests for all branches
def test_save10():
    assert apply_discount(100, "SAVE10") == 90

def test_save20():
    assert apply_discount(100, "SAVE20") == 80

def test_no_discount():
    assert apply_discount(100, "INVALID") == 100
```

### 4. Exclude Generated and Third-Party Code

```ini
# .coveragerc
[run]
omit =
    */migrations/*
    */venv/*
    */virtualenv/*
    */tests/*
    */node_modules/*
    */.tox/*
    */setup.py
```

```json
// Jest config
{
  "coveragePathIgnorePatterns": [
    "/node_modules/",
    "/dist/",
    "/coverage/",
    "\\.test\\.(js|ts)$"
  ]
}
```

### 5. Test Edge Cases

```python
def divide(a, b):
    if b == 0:
        raise ValueError("Division by zero")
    return a / b

# Test normal case
def test_divide():
    assert divide(10, 2) == 5

# Test edge case (improves coverage)
def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)

# Test other edge cases
def test_divide_negative():
    assert divide(-10, 2) == -5

def test_divide_floats():
    assert divide(1, 3) == pytest.approx(0.333, rel=0.01)
```

### 6. Measure Coverage Trends

```bash
# Track coverage over time
echo "$(date), $(coverage report | grep TOTAL | awk '{print $4}')" >> coverage_history.csv

# Fail if coverage decreases
previous_coverage=$(tail -1 coverage_history.csv | cut -d',' -f2)
current_coverage=$(coverage report | grep TOTAL | awk '{print $4}')

if (( $(echo "$current_coverage < $previous_coverage" | bc -l) )); then
    echo "Coverage decreased!"
    exit 1
fi
```

## Common Coverage Pitfalls

### 1. False Sense of Security

```python
# High coverage doesn't mean good tests
def add(a, b):
    return a + b

def test_add():
    # 100% coverage, but poor test
    add(2, 3)  # No assertion!
```

### 2. Testing Implementation, Not Behavior

```python
# Bad: Testing internal implementation
def test_user_creation():
    user = User("John")
    assert user._internal_id is not None  # Implementation detail
    assert user._created_timestamp is not None

# Good: Testing behavior
def test_user_creation():
    user = User("John")
    assert user.get_name() == "John"
    assert user.is_active() == True
```

### 3. Ignoring Uncovered Error Paths

```python
def process_payment(amount):
    if amount <= 0:
        raise ValueError("Invalid amount")  # Often not tested

    # Payment processing...
    return {"status": "success"}

# Add test for error path
def test_invalid_payment_amount():
    with pytest.raises(ValueError):
        process_payment(-100)
```

### 4. Over-Testing Trivial Code

```python
# Don't need tests for this
@property
def name(self):
    return self._name

@name.setter
def name(self, value):
    self._name = value
```

## Coverage for Different Languages

### Python

```bash
# pytest-cov
pytest --cov=myapp --cov-report=html

# coverage.py
coverage run -m pytest
coverage report
coverage html
```

### JavaScript/TypeScript

```bash
# Jest
npm test -- --coverage

# NYC
nyc npm test

# Mocha + NYC
nyc mocha test/**/*.js
```

### Java

```xml
<!-- Maven: JaCoCo -->
<plugin>
  <groupId>org.jacoco</groupId>
  <artifactId>jacoco-maven-plugin</artifactId>
  <executions>
    <execution>
      <goals>
        <goal>prepare-agent</goal>
      </goals>
    </execution>
    <execution>
      <id>report</id>
      <phase>test</phase>
      <goals>
        <goal>report</goal>
      </goals>
    </execution>
  </executions>
</plugin>
```

### Go

```bash
# Built-in coverage
go test -cover ./...

# Detailed coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

### Ruby

```ruby
# SimpleCov
require 'simplecov'
SimpleCov.start do
  add_filter '/spec/'
  add_filter '/config/'
end
```

## Realistic Coverage Targets

### General Guidelines

- **Overall project**: 70-80%
- **Critical business logic**: 90-100%
- **UI components**: 60-70%
- **Utilities**: 80-90%
- **Configuration**: 50-60%

### Example Targets

```json
{
  "coverageThreshold": {
    "global": {
      "branches": 75,
      "functions": 75,
      "lines": 75,
      "statements": 75
    },
    "src/core/**/*.js": {
      "branches": 90,
      "functions": 90,
      "lines": 90
    },
    "src/utils/**/*.js": {
      "branches": 85,
      "functions": 85,
      "lines": 85
    },
    "src/ui/**/*.js": {
      "branches": 65,
      "functions": 65,
      "lines": 65
    }
  }
}
```

## Quick Reference

### Commands

```bash
# Python
pytest --cov=myapp                        # Basic coverage
pytest --cov=myapp --cov-report=html      # HTML report
pytest --cov=myapp --cov-fail-under=80    # Fail if below 80%
coverage report -m                        # Show missing lines

# JavaScript (Jest)
npm test -- --coverage                    # Basic coverage
npm test -- --coverage --watch            # Watch mode

# NYC
nyc npm test                              # Basic coverage
nyc --reporter=html npm test              # HTML report
```

### Coverage Formulas

```
Line Coverage = (Executed Lines / Total Lines) × 100%
Branch Coverage = (Executed Branches / Total Branches) × 100%
Function Coverage = (Called Functions / Total Functions) × 100%
```

## Further Resources

- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Jest Coverage](https://jestjs.io/docs/cli#--coverageboolean)
- [Istanbul/NYC](https://istanbul.js.org/)
- [Codecov](https://about.codecov.io/)
- [Martin Fowler - Test Coverage](https://martinfowler.com/bliki/TestCoverage.html)
