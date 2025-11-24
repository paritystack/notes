# Code Quality

Code quality tools help maintain consistent, readable, and error-free code through linting, formatting, type checking, and static analysis.

## Overview

**Code Quality Tools:**
- **Linters**: Find bugs, style issues, and suspicious code
- **Formatters**: Automatically format code consistently
- **Type Checkers**: Catch type-related errors
- **Static Analyzers**: Deep code analysis for bugs and vulnerabilities

## Linting

### Python: Flake8

**Installation:**
```bash
pip install flake8
```

**Usage:**
```bash
# Lint all Python files
flake8

# Lint specific files
flake8 myapp/

# Show statistics
flake8 --statistics

# Output to file
flake8 --output-file=report.txt
```

**Configuration (.flake8):**
```ini
[flake8]
max-line-length = 88
exclude =
    .git,
    __pycache__,
    venv,
    .venv,
    migrations
ignore =
    E203,  # Whitespace before ':'
    E501,  # Line too long (handled by black)
    W503   # Line break before binary operator
per-file-ignores =
    __init__.py:F401
max-complexity = 10
```

### Python: Pylint

**Installation:**
```bash
pip install pylint
```

**Usage:**
```bash
# Lint files
pylint myapp/

# Generate config
pylint --generate-rcfile > .pylintrc

# Show only errors
pylint --errors-only myapp/

# Disable specific warnings
pylint --disable=C0111,R0903 myapp/
```

**Configuration (.pylintrc):**
```ini
[MASTER]
ignore=migrations,venv,.venv

[MESSAGES CONTROL]
disable=
    C0111,  # missing-docstring
    C0103,  # invalid-name
    R0903,  # too-few-public-methods

[FORMAT]
max-line-length=88

[DESIGN]
max-args=7
max-attributes=10
```

### Python: Ruff

Modern, fast Python linter (10-100x faster than Flake8).

**Installation:**
```bash
pip install ruff
```

**Usage:**
```bash
# Lint
ruff check .

# Auto-fix
ruff check --fix .

# Watch mode
ruff check --watch .
```

**Configuration (pyproject.toml):**
```toml
[tool.ruff]
line-length = 88
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "W",   # pycodestyle warnings
]
ignore = [
    "E501",  # line too long
]
exclude = [
    ".git",
    "__pycache__",
    "venv",
    "migrations",
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
```

### JavaScript: ESLint

**Installation:**
```bash
npm install --save-dev eslint

# Initialize config
npx eslint --init
```

**Usage:**
```bash
# Lint files
npx eslint src/

# Auto-fix
npx eslint src/ --fix

# Specific file
npx eslint src/index.js

# Output format
npx eslint src/ --format json
```

**Configuration (.eslintrc.js):**
```javascript
module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:@typescript-eslint/recommended',
  ],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
  },
  plugins: ['react', '@typescript-eslint'],
  rules: {
    'no-console': 'warn',
    'no-unused-vars': 'error',
    'prefer-const': 'error',
    'semi': ['error', 'always'],
    'quotes': ['error', 'single'],
  },
  ignorePatterns: [
    'node_modules/',
    'dist/',
    'build/',
  ],
};
```

**Popular presets:**
```bash
# Airbnb style guide
npm install --save-dev eslint-config-airbnb

# Standard JS
npm install --save-dev eslint-config-standard

# Prettier integration
npm install --save-dev eslint-config-prettier
```

## Code Formatting

### Python: Black

The uncompromising Python code formatter.

**Installation:**
```bash
pip install black
```

**Usage:**
```bash
# Format all files
black .

# Check without modifying
black --check .

# Show diff
black --diff .

# Format specific files
black myapp/

# Exclude files
black . --exclude venv
```

**Configuration (pyproject.toml):**
```toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
  | \.venv
  | venv
  | migrations
  | __pycache__
)/
'''
```

**Before:**
```python
def   function(  arg1,arg2,  arg3  ):
    x=1+2+3
    y={'key':  'value'}
    return x,y
```

**After:**
```python
def function(arg1, arg2, arg3):
    x = 1 + 2 + 3
    y = {"key": "value"}
    return x, y
```

### JavaScript: Prettier

**Installation:**
```bash
npm install --save-dev prettier
```

**Usage:**
```bash
# Format all files
npx prettier --write .

# Check formatting
npx prettier --check .

# Specific files
npx prettier --write "src/**/*.{js,jsx,ts,tsx}"
```

**Configuration (.prettierrc):**
```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "arrowParens": "avoid",
  "endOfLine": "lf"
}
```

**Ignore files (.prettierignore):**
```
node_modules/
dist/
build/
coverage/
*.min.js
```

## Type Checking

### Python: mypy

Static type checker for Python.

**Installation:**
```bash
pip install mypy
```

**Usage:**
```bash
# Type check files
mypy myapp/

# Strict mode
mypy --strict myapp/

# Show error codes
mypy --show-error-codes myapp/

# Ignore missing imports
mypy --ignore-missing-imports myapp/
```

**Type annotations:**
```python
from typing import List, Dict, Optional, Union

def greet(name: str) -> str:
    return f"Hello, {name}"

def process_items(items: List[int]) -> int:
    return sum(items)

def get_user(id: int) -> Optional[Dict[str, str]]:
    if id > 0:
        return {"name": "John", "email": "john@example.com"}
    return None

def calculate(x: Union[int, float]) -> float:
    return x * 2.5
```

**Configuration (mypy.ini):**
```ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
ignore_missing_imports = True

[mypy-tests.*]
ignore_errors = True
```

### TypeScript

**Installation:**
```bash
npm install --save-dev typescript
```

**Configuration (tsconfig.json):**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noImplicitReturns": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

**Usage:**
```bash
# Type check
npx tsc --noEmit

# Watch mode
npx tsc --watch

# Build
npx tsc
```

**Type examples:**
```typescript
// Basic types
let name: string = "John";
let age: number = 30;
let active: boolean = true;

// Arrays
let numbers: number[] = [1, 2, 3];
let names: Array<string> = ["John", "Jane"];

// Objects
interface User {
  name: string;
  age: number;
  email?: string;  // Optional
}

const user: User = {
  name: "John",
  age: 30
};

// Functions
function greet(name: string): string {
  return `Hello, ${name}`;
}

// Generics
function identity<T>(arg: T): T {
  return arg;
}

// Union types
let value: string | number;
value = "text";
value = 123;
```

## Static Analysis

### Python: Bandit

Security-focused static analyzer.

**Installation:**
```bash
pip install bandit
```

**Usage:**
```bash
# Scan for security issues
bandit -r myapp/

# Generate report
bandit -r myapp/ -f json -o report.json

# Severity levels
bandit -r myapp/ -ll  # Low severity and above
bandit -r myapp/ -lll # Only high severity
```

**Configuration (.bandit):**
```yaml
exclude_dirs:
  - /test
  - /venv
tests:
  - B201
  - B301
  - B302
```

### JavaScript: SonarQube

**Installation:**
```bash
npm install --save-dev sonarqube-scanner
```

**Configuration (sonar-project.properties):**
```properties
sonar.projectKey=my-project
sonar.sources=src
sonar.tests=tests
sonar.javascript.lcov.reportPaths=coverage/lcov.info
```

### Security Scanning

**npm audit:**
```bash
# Check for vulnerabilities
npm audit

# Fix vulnerabilities
npm audit fix

# Force fix (may break)
npm audit fix --force
```

**pip-audit:**
```bash
pip install pip-audit

# Scan dependencies
pip-audit

# Auto-fix
pip-audit --fix
```

## Pre-commit Hooks

Automatically run checks before commits.

### Python: pre-commit

**Installation:**
```bash
pip install pre-commit
```

**Configuration (.pre-commit-config.yaml):**
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
```

**Installation:**
```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

### JavaScript: Husky + lint-staged

**Installation:**
```bash
npm install --save-dev husky lint-staged
npx husky install
```

**Configuration (package.json):**
```json
{
  "scripts": {
    "prepare": "husky install"
  },
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,css}": [
      "prettier --write"
    ]
  }
}
```

**Create hook:**
```bash
npx husky add .husky/pre-commit "npx lint-staged"
```

## Import Sorting

### Python: isort

**Installation:**
```bash
pip install isort
```

**Usage:**
```bash
# Sort imports
isort .

# Check only
isort --check-only .

# Show diff
isort --diff .
```

**Configuration (pyproject.toml):**
```toml
[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

**Before:**
```python
import os
from myapp import utils
import sys
from typing import List
from myapp.models import User
import json
```

**After:**
```python
import json
import os
import sys
from typing import List

from myapp import utils
from myapp.models import User
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install black flake8 mypy isort

      - name: Black formatting check
        run: black --check .

      - name: Flake8 linting
        run: flake8 .

      - name: isort import sorting
        run: isort --check-only .

      - name: mypy type checking
        run: mypy myapp/

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install Node dependencies
        run: npm ci

      - name: ESLint
        run: npm run lint

      - name: Prettier check
        run: npx prettier --check .

      - name: TypeScript check
        run: npx tsc --noEmit
```

### GitLab CI

```yaml
quality:
  stage: test
  image: python:3.11
  before_script:
    - pip install black flake8 mypy isort
  script:
    - black --check .
    - flake8 .
    - isort --check-only .
    - mypy myapp/
  only:
    - merge_requests
    - main
```

## Editor Integration

### VS Code

**Python extensions:**
- Python (Microsoft)
- Pylance
- Black Formatter
- Flake8
- isort

**Settings (.vscode/settings.json):**
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.pylintEnabled": false,
  "python.formatting.provider": "black",
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact"
  ],
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  }
}
```

### PyCharm / IntelliJ

**Enable tools:**
1. Settings → Tools → Black
2. Settings → Editor → Inspections → Python
3. Settings → Tools → External Tools

## Best Practices

### 1. Consistent Configuration

Use configuration files committed to repository:
- `.flake8`, `pyproject.toml` for Python
- `.eslintrc.js`, `.prettierrc` for JavaScript
- Share settings across team

### 2. Automate Formatting

Don't argue about style:
```bash
# Format on save in editor
# Run formatters in pre-commit hooks
# Enforce in CI/CD
```

### 3. Progressive Enhancement

Start with basics, add more:
```bash
# Week 1: Add formatter
# Week 2: Add linter
# Week 3: Add type checking
# Week 4: Add security scanning
```

### 4. Fix Issues, Don't Ignore

```python
# Bad: Ignoring everywhere
# flake8: noqa
# pylint: disable=all

# Good: Fix the issue
def calculate_total(items):  # Was: def calculateTotal
    return sum(item.price for item in items)
```

### 5. Fail Fast

```yaml
# CI: Fail on quality issues
- name: Check code quality
  run: |
    black --check . || exit 1
    flake8 . || exit 1
    mypy . || exit 1
```

## Common Configuration

### Python Project

**pyproject.toml:**
```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### JavaScript Project

**package.json:**
```json
{
  "scripts": {
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix",
    "format": "prettier --write .",
    "format:check": "prettier --check .",
    "type-check": "tsc --noEmit",
    "quality": "npm run lint && npm run format:check && npm run type-check"
  },
  "devDependencies": {
    "eslint": "^8.0.0",
    "prettier": "^3.0.0",
    "typescript": "^5.0.0",
    "husky": "^8.0.0",
    "lint-staged": "^13.0.0"
  }
}
```

## Quick Reference

### Python Tools

| Tool | Purpose | Command |
|------|---------|---------|
| Black | Formatting | `black .` |
| Flake8 | Linting | `flake8 .` |
| Pylint | Linting | `pylint myapp/` |
| Ruff | Fast linting | `ruff check .` |
| isort | Import sorting | `isort .` |
| mypy | Type checking | `mypy myapp/` |
| Bandit | Security | `bandit -r myapp/` |

### JavaScript Tools

| Tool | Purpose | Command |
|------|---------|---------|
| Prettier | Formatting | `prettier --write .` |
| ESLint | Linting | `eslint src/` |
| TypeScript | Type checking | `tsc --noEmit` |
| npm audit | Security | `npm audit` |

## Further Resources

- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [ESLint Rules](https://eslint.org/docs/rules/)
- [Prettier Options](https://prettier.io/docs/en/options.html)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/intro.html)
- [mypy Documentation](https://mypy.readthedocs.io/)
