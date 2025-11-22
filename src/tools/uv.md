# uv

An extremely fast Python package and project manager, written in Rust.

## Overview

uv is a modern Python package manager designed to replace pip, pip-tools, pipx, poetry, pyenv, and virtualenv with a single, unified tool. Created by Astral (the team behind ruff), uv is 10-100x faster than traditional Python package managers while maintaining compatibility with existing Python packaging standards.

uv handles dependency resolution, virtual environment management, Python version installation, and project scaffolding. It uses a global cache to minimize redundant downloads and supports both pyproject.toml-based projects and ad-hoc package installations. The tool is designed to be a drop-in replacement for existing workflows while offering significant performance improvements.

**Key Features:**
- **Blazing Fast**: 10-100x faster than pip and pip-tools due to Rust implementation
- **Python Version Management**: Install and manage multiple Python versions
- **Universal Resolver**: Advanced dependency resolution with backtracking
- **Lock Files**: Reproducible builds with uv.lock
- **Workspace Support**: Manage monorepos and multi-package projects
- **Drop-in Replacement**: Compatible with pip, pip-tools, and virtualenv workflows
- **Global Cache**: Deduplicated package storage across projects
- **Offline Mode**: Work without internet access using cached packages

## Installation

### Using the standalone installer (recommended):

```bash
# Linux and macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Using pip:

```bash
pip install uv
```

### Using pipx:

```bash
pipx install uv
```

### Using Homebrew (macOS):

```bash
brew install uv
```

### Using cargo:

```bash
cargo install uv
```

### Verify installation:

```bash
uv --version
# Example output: uv 0.4.0
```

## Basic Usage

### Initialize a New Project

```bash
# Create a new Python project
uv init my-project
cd my-project

# Creates:
# - pyproject.toml (project configuration)
# - README.md
# - .python-version (Python version specification)
# - src/my_project/ (source code directory)

# Initialize in existing directory
uv init
```

### Add Dependencies

```bash
# Add a package
uv add requests

# Add multiple packages
uv add requests httpx pandas

# Add a development dependency
uv add --dev pytest black ruff

# Add with version constraint
uv add "django>=4.0,<5.0"
uv add "numpy~=1.24.0"

# Add from a git repository
uv add git+https://github.com/user/repo.git

# Add from a specific branch or tag
uv add git+https://github.com/user/repo.git@main
uv add git+https://github.com/user/repo.git@v1.0.0
```

### Remove Dependencies

```bash
# Remove a package
uv remove requests

# Remove multiple packages
uv remove requests httpx
```

### Install Dependencies

```bash
# Install all dependencies from pyproject.toml
uv sync

# Install only production dependencies
uv sync --no-dev

# Install dependencies and create/update lock file
uv lock
uv sync
```

### Run Commands

```bash
# Run a Python script
uv run script.py

# Run a module
uv run -m pytest

# Run with specific Python version
uv run --python 3.12 script.py

# Run an inline script
uv run --with requests - <<EOF
import requests
response = requests.get("https://api.github.com")
print(response.json())
EOF
```

### Install Packages (pip-compatible)

```bash
# Install a package (like pip install)
uv pip install requests

# Install from requirements.txt
uv pip install -r requirements.txt

# Install with extras
uv pip install "fastapi[all]"

# Install in editable mode
uv pip install -e .
```

## Configuration

### pyproject.toml

uv uses the standard `pyproject.toml` file for project configuration:

```toml
[project]
name = "my-project"
version = "0.1.0"
description = "A sample project"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.31.0",
    "httpx>=0.24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
# uv-specific configuration
dev-dependencies = [
    "pytest>=7.0.0",
    "mypy>=1.0.0",
]

[tool.uv.sources]
# Specify custom package sources
my-package = { git = "https://github.com/user/repo.git" }
```

### Global Configuration (uv.toml)

Create `~/.config/uv/uv.toml` or `uv.toml` in your project:

```toml
# Set default Python version
python-version = "3.12"

# Configure cache location
cache-dir = "/custom/cache/path"

# Set index URL
index-url = "https://pypi.org/simple"

# Add extra index URLs
extra-index-url = ["https://custom-index.com/simple"]

# Configure trusted hosts
trusted-host = ["custom-index.com"]

# Offline mode
offline = false

# Link mode (hardlink, copy, or symlink)
link-mode = "copy"

# Concurrency settings
concurrent-downloads = 50
concurrent-builds = 4
```

### Environment Variables

```bash
# Set Python version
export UV_PYTHON="3.12"

# Set cache directory
export UV_CACHE_DIR="/custom/cache/path"

# Set index URL
export UV_INDEX_URL="https://pypi.org/simple"

# Enable offline mode
export UV_OFFLINE=1

# Set system Python (for uv pip)
export UV_SYSTEM_PYTHON=1

# Configure link mode
export UV_LINK_MODE="copy"
```

## Virtual Environments

### Creating Virtual Environments

```bash
# Create a virtual environment
uv venv

# Create with specific name
uv venv .venv

# Create with specific Python version
uv venv --python 3.12
uv venv --python python3.11

# Create with system site packages
uv venv --system-site-packages

# Create without pip
uv venv --no-pip
```

### Activating Virtual Environments

```bash
# Linux/macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat
```

### Using Virtual Environments

```bash
# uv automatically detects and uses .venv
uv pip install requests

# Specify a different virtual environment
uv pip install --python .venv/bin/python requests

# Use system Python (no virtual environment)
uv pip install --system requests
```

## Package Management

### Installing Packages

```bash
# Install latest version
uv pip install requests

# Install specific version
uv pip install requests==2.31.0

# Install with version constraints
uv pip install "requests>=2.28,<3.0"
uv pip install "requests~=2.31.0"

# Install with extras
uv pip install "fastapi[all]"
uv pip install "django[bcrypt,argon2]"

# Install from requirements file
uv pip install -r requirements.txt
uv pip install -r requirements.in

# Install from URL
uv pip install https://github.com/user/repo/archive/main.zip

# Install local package
uv pip install .
uv pip install -e .  # Editable mode
```

### Updating Packages

```bash
# Upgrade a package
uv pip install --upgrade requests

# Upgrade all packages
uv pip install --upgrade -r requirements.txt

# Update lock file with latest versions
uv lock --upgrade

# Upgrade specific package in lock file
uv lock --upgrade-package requests
```

### Listing Packages

```bash
# List installed packages
uv pip list

# List in requirements format
uv pip freeze

# Show package details
uv pip show requests
```

### Uninstalling Packages

```bash
# Uninstall a package
uv pip uninstall requests

# Uninstall multiple packages
uv pip uninstall requests httpx pandas

# Uninstall from requirements file
uv pip uninstall -r requirements.txt
```

### Compiling Requirements

```bash
# Generate lock file from requirements.in
uv pip compile requirements.in -o requirements.txt

# Compile with specific Python version
uv pip compile --python-version 3.12 requirements.in

# Compile with extras
uv pip compile --extra dev requirements.in

# Upgrade all packages during compile
uv pip compile --upgrade requirements.in

# Upgrade specific package
uv pip compile --upgrade-package requests requirements.in
```

### Syncing Environment

```bash
# Sync environment to match requirements.txt
uv pip sync requirements.txt

# Sync with dev dependencies
uv pip sync requirements.txt dev-requirements.txt

# Strict sync (uninstall packages not in requirements)
uv pip sync --strict requirements.txt
```

## Python Version Management

### Installing Python Versions

```bash
# Install latest Python version
uv python install

# Install specific version
uv python install 3.12
uv python install 3.11.5

# Install multiple versions
uv python install 3.11 3.12

# List available Python versions
uv python list --all-versions

# List installed Python versions
uv python list
```

### Using Python Versions

```bash
# Set Python version for project
echo "3.12" > .python-version

# Use specific Python version
uv run --python 3.12 script.py
uv venv --python 3.11

# Pin Python version in pyproject.toml
# requires-python = ">=3.11"
```

### Managing Python Installations

```bash
# Find Python installations
uv python find

# Find specific version
uv python find 3.12

# Uninstall Python version
uv python uninstall 3.11.5

# Show Python installation directory
uv python dir
```

## Lock Files

### Creating and Using Lock Files

```bash
# Generate uv.lock from pyproject.toml
uv lock

# Install from lock file
uv sync

# Update lock file with latest versions
uv lock --upgrade

# Update specific package in lock file
uv lock --upgrade-package requests

# Install without updating lock file
uv sync --frozen
```

### Lock File Structure

The `uv.lock` file contains resolved dependencies with exact versions and hashes:

```toml
version = 1
requires-python = ">=3.11"

[[package]]
name = "requests"
version = "2.31.0"
source = { registry = "https://pypi.org/simple" }
dependencies = [
    { name = "certifi" },
    { name = "charset-normalizer" },
    { name = "idna" },
    { name = "urllib3" },
]
wheels = [
    { url = "https://...", hash = "sha256:..." },
]

[[package]]
name = "certifi"
version = "2024.2.2"
source = { registry = "https://pypi.org/simple" }
wheels = [
    { url = "https://...", hash = "sha256:..." },
]
```

### Lock File Best Practices

```bash
# Commit uv.lock to version control
git add uv.lock

# Use frozen mode in CI/CD
uv sync --frozen

# Update dependencies regularly
uv lock --upgrade

# Check for outdated packages
uv pip list --outdated
```

## Workspaces

### Creating a Workspace

Workspaces allow managing multiple related packages in a single repository:

```toml
# pyproject.toml (root)
[tool.uv.workspace]
members = [
    "packages/*",
    "apps/*",
]

[tool.uv]
dev-dependencies = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
]
```

### Workspace Structure

```
my-workspace/
├── pyproject.toml          # Workspace root
├── packages/
│   ├── package-a/
│   │   ├── pyproject.toml
│   │   └── src/
│   └── package-b/
│       ├── pyproject.toml
│       └── src/
└── apps/
    └── my-app/
        ├── pyproject.toml
        └── src/
```

### Working with Workspaces

```bash
# Install all workspace members
uv sync

# Add dependency to specific member
cd packages/package-a
uv add requests

# Add workspace dependency
# In package-b/pyproject.toml:
# dependencies = ["package-a"]

# Run commands in workspace context
uv run --package package-a pytest
```

## Advanced Features

### Custom Package Indices

```bash
# Use custom index
uv pip install --index-url https://custom-index.com/simple requests

# Add extra index
uv pip install --extra-index-url https://custom-index.com/simple requests

# Use custom index in pyproject.toml
# [tool.uv]
# index-url = "https://custom-index.com/simple"
# extra-index-url = ["https://pypi.org/simple"]
```

### Authentication

```bash
# Use credentials in URL
uv pip install --index-url https://user:password@custom-index.com/simple package

# Use environment variables
export UV_INDEX_URL="https://user:password@custom-index.com/simple"

# Use keyring for credentials
uv pip install --keyring-provider subprocess package

# Configure in pyproject.toml with credential helpers
# [tool.uv]
# index-url = "https://custom-index.com/simple"
```

### Cache Management

```bash
# Show cache directory
uv cache dir

# Show cache size
uv cache size

# Clean cache
uv cache clean

# Clean specific package
uv cache clean requests

# Prune cache (remove unused entries)
uv cache prune

# Disable cache for a command
UV_NO_CACHE=1 uv pip install requests
```

### Resolution Strategies

```bash
# Use highest version (default)
uv pip install requests

# Use lowest compatible version
uv pip install --resolution lowest requests

# Use lowest-direct (lowest for direct deps, highest for transitive)
uv pip install --resolution lowest-direct requests

# Specify resolution strategy in pyproject.toml
# [tool.uv]
# resolution = "highest"
```

### Platform-Specific Dependencies

```toml
[project]
dependencies = [
    "requests>=2.31.0",
]

[project.optional-dependencies]
# Platform markers
windows = [
    "pywin32>=305; sys_platform == 'win32'",
]
linux = [
    "uvloop>=0.19.0; sys_platform == 'linux'",
]

# Python version markers
py312 = [
    "tomli>=2.0.0; python_version < '3.11'",
]
```

### Build Isolation

```bash
# Disable build isolation
uv pip install --no-build-isolation package

# Install build dependencies manually
uv pip install setuptools wheel
uv pip install --no-build-isolation -e .
```

### Trusted Hosts

```bash
# Trust a host (skip TLS verification)
uv pip install --trusted-host custom-index.com package

# Configure in pyproject.toml
# [tool.uv]
# trusted-host = ["custom-index.com"]
```

## Best Practices

1. **Use Lock Files for Reproducibility**
   ```bash
   # Always commit uv.lock to version control
   git add uv.lock

   # Use frozen mode in CI/CD
   uv sync --frozen
   ```

2. **Leverage the Global Cache**
   ```bash
   # Let uv manage the cache automatically
   # Cache is shared across all projects
   # No need to download packages multiple times
   ```

3. **Pin Python Versions**
   ```bash
   # Use .python-version for consistency
   echo "3.12" > .python-version

   # Or specify in pyproject.toml
   # requires-python = ">=3.11,<4.0"
   ```

4. **Organize Dependencies**
   ```toml
   # Use optional dependencies for dev tools
   [project.optional-dependencies]
   dev = ["pytest", "black", "ruff"]
   docs = ["mkdocs", "mkdocs-material"]
   ```

5. **Use Workspaces for Monorepos**
   ```bash
   # Manage related packages together
   # Share dev dependencies across packages
   # Simplify inter-package dependencies
   ```

6. **Keep Dependencies Updated**
   ```bash
   # Regularly update lock file
   uv lock --upgrade

   # Check for outdated packages
   uv pip list --outdated
   ```

7. **Use Version Constraints Wisely**
   ```toml
   # Be specific but not too restrictive
   dependencies = [
       "requests>=2.31,<3.0",  # Good
       "httpx~=0.24.0",        # Allows patch updates
       "pandas>=2.0",          # Allow minor updates
   ]
   ```

8. **Leverage uv run for Scripts**
   ```bash
   # No need to activate virtual environments
   uv run script.py
   uv run -m pytest
   ```

9. **Use Requirements Files for CI/CD**
   ```bash
   # Generate requirements.txt for compatibility
   uv pip compile pyproject.toml -o requirements.txt

   # Or use uv directly in CI
   uv sync --frozen
   ```

10. **Configure Resolution Strategy**
    ```toml
    # Use lowest-direct for libraries
    [tool.uv]
    resolution = "lowest-direct"

    # Ensures compatibility with older versions
    ```

## Common Patterns

### Setting Up a New Project

```bash
# Create and initialize project
uv init my-project
cd my-project

# Add dependencies
uv add fastapi uvicorn sqlalchemy pydantic

# Add dev dependencies
uv add --dev pytest pytest-cov black ruff mypy

# Create lock file
uv lock

# Run the application
uv run uvicorn main:app --reload
```

### Converting from pip/requirements.txt

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create pyproject.toml from requirements.txt
uv init

# Add existing dependencies
cat requirements.txt | xargs uv add

# Or use pip compatibility mode
uv pip install -r requirements.txt
uv pip freeze > uv-requirements.txt
```

### Converting from Poetry

```bash
# Poetry uses pyproject.toml, so it's mostly compatible

# Install dependencies with uv
uv sync

# Replace poetry commands:
# poetry add pkg -> uv add pkg
# poetry install -> uv sync
# poetry run cmd -> uv run cmd
# poetry shell -> source .venv/bin/activate

# Note: You may need to adjust some poetry-specific configurations
```

### CI/CD Pipeline

```bash
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run tests
        run: uv run pytest

      - name: Run linting
        run: |
          uv run ruff check .
          uv run black --check .
```

### Docker Integration

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Run application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0"]
```

### Managing Multiple Python Versions

```bash
# Install multiple Python versions
uv python install 3.11 3.12 3.13

# Test against multiple versions
for version in 3.11 3.12 3.13; do
  echo "Testing with Python $version"
  uv run --python $version pytest
done

# Use tox with uv
# tox.ini
# [tox]
# envlist = py311,py312,py313
#
# [testenv]
# runner = uv-venv-runner
# commands = pytest
```

### Scripting with uv

```bash
# Create standalone script with dependencies
cat > script.py << 'EOF'
# /// script
# dependencies = [
#   "requests",
#   "rich",
# ]
# ///

import requests
from rich import print

response = requests.get("https://api.github.com")
print(response.json())
EOF

# Run script (uv installs deps automatically)
uv run script.py
```

### Private Package Repository

```toml
# pyproject.toml
[tool.uv]
index-url = "https://pypi.org/simple"
extra-index-url = [
    "https://private.pypi.company.com/simple",
]

[[tool.uv.index]]
name = "private"
url = "https://private.pypi.company.com/simple"
explicit = true  # Only use when explicitly specified

[project]
dependencies = [
    "public-package>=1.0",
    "private-package>=2.0 @ private",  # From private index
]
```

## Troubleshooting

### Installation Issues

**Problem**: uv command not found after installation
```bash
# Solution: Add uv to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or reload shell configuration
source ~/.bashrc  # or ~/.zshrc
```

**Problem**: Permission denied when installing
```bash
# Solution: Install in user space
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip with --user flag
pip install --user uv
```

### Dependency Resolution

**Problem**: No solution found during resolution
```bash
# Solution 1: Check for conflicting requirements
uv pip install --verbose package

# Solution 2: Try different resolution strategy
uv pip install --resolution lowest package

# Solution 3: Temporarily relax version constraints
uv add "package>=1.0"  # Instead of ==1.0.0
```

**Problem**: Incompatible Python version
```bash
# Solution: Install compatible Python version
uv python install 3.12

# Or update requires-python in pyproject.toml
# requires-python = ">=3.11"
```

### Virtual Environment Issues

**Problem**: Wrong Python version in virtual environment
```bash
# Solution: Recreate venv with specific version
rm -rf .venv
uv venv --python 3.12
```

**Problem**: Packages not found after installation
```bash
# Solution: Ensure virtual environment is activated
source .venv/bin/activate

# Or use uv run
uv run python -c "import requests"
```

### Cache Problems

**Problem**: Corrupted cache
```bash
# Solution: Clean and rebuild cache
uv cache clean
uv sync --refresh
```

**Problem**: Disk space issues
```bash
# Solution: Prune unused cache entries
uv cache prune

# Check cache size
uv cache size
```

### Network Issues

**Problem**: SSL certificate verification failed
```bash
# Solution 1: Update certificates
pip install --upgrade certifi

# Solution 2: Trust host (not recommended for production)
uv pip install --trusted-host pypi.org package
```

**Problem**: Timeout during package download
```bash
# Solution: Increase timeout
UV_HTTP_TIMEOUT=300 uv pip install package

# Or use offline mode if packages are cached
UV_OFFLINE=1 uv pip install package
```

### Lock File Issues

**Problem**: Lock file out of sync with pyproject.toml
```bash
# Solution: Regenerate lock file
uv lock

# Or force sync
uv sync --refresh
```

**Problem**: Merge conflicts in uv.lock
```bash
# Solution: Regenerate from pyproject.toml
git checkout --theirs pyproject.toml
uv lock
```

### Platform-Specific Issues

**Problem**: Package only available for certain platforms
```bash
# Solution: Use platform markers
[project]
dependencies = [
    "pywin32>=305; sys_platform == 'win32'",
    "uvloop>=0.19.0; sys_platform != 'win32'",
]
```

**Problem**: Build failures on Windows
```bash
# Solution: Install Visual C++ Build Tools
# Or use pre-built wheels
uv pip install --only-binary :all: package
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `uv init` | Initialize a new project |
| `uv add <package>` | Add a dependency |
| `uv remove <package>` | Remove a dependency |
| `uv sync` | Install dependencies from pyproject.toml/lock file |
| `uv lock` | Generate or update lock file |
| `uv run <script>` | Run a Python script |
| `uv run -m <module>` | Run a Python module |
| `uv pip install <package>` | Install a package (pip-compatible) |
| `uv pip uninstall <package>` | Uninstall a package |
| `uv pip list` | List installed packages |
| `uv pip freeze` | Output installed packages in requirements format |
| `uv pip compile` | Compile requirements file with resolved versions |
| `uv pip sync` | Sync environment to requirements file |
| `uv venv` | Create a virtual environment |
| `uv python install <version>` | Install a Python version |
| `uv python list` | List available Python versions |
| `uv cache clean` | Clear the package cache |
| `uv cache prune` | Remove unused cache entries |
| `uv cache dir` | Show cache directory |
| `uv --version` | Show uv version |
| `uv --help` | Show help information |

## Additional Options

| Option | Description |
|--------|-------------|
| `--python <version>` | Specify Python version to use |
| `--no-cache` | Disable cache for this operation |
| `--offline` | Work in offline mode |
| `--quiet` | Minimize output |
| `--verbose` | Show detailed output |
| `--frozen` | Use exact versions from lock file |
| `--no-dev` | Exclude development dependencies |
| `--upgrade` | Upgrade packages to latest versions |
| `--resolution <strategy>` | Set resolution strategy (highest, lowest, lowest-direct) |
| `--index-url <url>` | Set package index URL |
| `--extra-index-url <url>` | Add additional package index |

uv represents the future of Python package management, combining the speed of Rust with the flexibility of Python's ecosystem to deliver a superior developer experience.
