# mdBook

mdBook is a command-line tool for creating books from Markdown files, similar to Gitbook but implemented in Rust. It's fast, simple, and ideal for technical documentation, tutorials, and books.

## Overview

mdBook takes Markdown files and generates a static website with built-in search, syntax highlighting, and theme support. It's the tool used to create the official Rust programming language book.

**Key Features:**
- Fast static site generation
- Automatic table of contents
- Built-in search functionality
- Syntax highlighting for code
- Light and dark themes
- Live preview with hot reloading
- Markdown extensions
- Customizable with preprocessors

## Installation

```bash
# Using Cargo (Rust package manager)
cargo install mdbook

# Ubuntu/Debian (from binary)
wget https://github.com/rust-lang/mdBook/releases/download/v0.4.36/mdbook-v0.4.36-x86_64-unknown-linux-gnu.tar.gz
tar xzf mdbook-v0.4.36-x86_64-unknown-linux-gnu.tar.gz
sudo mv mdbook /usr/local/bin/

# macOS
brew install mdbook

# From source
git clone https://github.com/rust-lang/mdBook.git
cd mdBook
cargo build --release
sudo cp target/release/mdbook /usr/local/bin/

# Verify installation
mdbook --version
```

## Quick Start

### Create a New Book

```bash
# Create new book
mdbook init mybook

# Project structure created:
# mybook/
# ├── book.toml       # Configuration file
# └── src/
#     ├── SUMMARY.md  # Table of contents
#     └── chapter_1.md

# Enter directory
cd mybook

# Build the book
mdbook build

# Serve with live preview
mdbook serve

# Open in browser
open http://localhost:3000
```

### Project Structure

```
mybook/
├── book.toml              # Configuration
├── src/
│   ├── SUMMARY.md         # Table of contents (required)
│   ├── chapter_1.md       # Chapter files
│   ├── chapter_2.md
│   ├── images/            # Images directory
│   │   └── diagram.png
│   └── sub_chapter/
│       └── section.md
└── book/                  # Generated output (git ignore)
    ├── index.html
    ├── chapter_1.html
    └── ...
```

## Configuration

### Basic book.toml

```toml
[book]
title = "My Amazing Book"
authors = ["John Doe"]
language = "en"
multilingual = false
src = "src"

[build]
build-dir = "book"
create-missing = true

[output.html]
default-theme = "light"
preferred-dark-theme = "navy"
git-repository-url = "https://github.com/user/repo"
git-repository-icon = "fa-github"
```

### Advanced Configuration

```toml
[book]
title = "Advanced Guide"
authors = ["Jane Smith", "John Doe"]
description = "A comprehensive guide"
language = "en"
multilingual = false
src = "src"

[build]
build-dir = "book"
create-missing = true

[preprocessor.links]

[output.html]
# Theme
default-theme = "rust"
preferred-dark-theme = "navy"
curly-quotes = true

# Repository
git-repository-url = "https://github.com/user/repo"
git-repository-icon = "fa-github"

# Navigation
additional-css = ["custom.css"]
additional-js = ["custom.js"]

# Code
no-section-label = false

# Search
[output.html.search]
enable = true
limit-results = 30
teaser-word-count = 30
use-boolean-and = true
boost-title = 2
boost-hierarchy = 1
boost-paragraph = 1
expand = true
heading-split-level = 3

# Print
[output.html.print]
enable = true

# Playground (for Rust code)
[output.html.playground]
editable = true
copyable = true
copy-js = true
line-numbers = false
runnable = true
```

## SUMMARY.md Format

### Basic Structure

```markdown
# Summary

[Introduction](./introduction.md)

# User Guide

- [Getting Started](./guide/getting-started.md)
- [Installation](./guide/installation.md)
    - [Linux](./guide/installation/linux.md)
    - [macOS](./guide/installation/macos.md)
    - [Windows](./guide/installation/windows.md)
- [Configuration](./guide/configuration.md)

# Reference

- [API Reference](./reference/api.md)
- [CLI Commands](./reference/cli.md)

# Appendix

- [Glossary](./appendix/glossary.md)
- [Contributors](./appendix/contributors.md)
```

### Advanced Features

```markdown
# Summary

[Preface](./preface.md)

---

# Part I: Basics

- [Chapter 1](./chapter-1.md)
- [Chapter 2](./chapter-2.md)

---

# Part II: Advanced

- [Chapter 3](./chapter-3.md)
    - [Section 3.1](./chapter-3/section-1.md)
    - [Section 3.2](./chapter-3/section-2.md)

---

[Conclusion](./conclusion.md)
[Appendix](./appendix.md)
```

## Commands

### Build Commands

```bash
# Build book
mdbook build

# Build and watch for changes
mdbook watch

# Serve with live reload
mdbook serve

# Serve on different port
mdbook serve -p 8080

# Serve on specific address
mdbook serve -n 0.0.0.0

# Open in browser
mdbook serve --open

# Build to different directory
mdbook build -d /tmp/mybook
```

### Testing

```bash
# Test code examples
mdbook test

# Test with specific library
mdbook test --library-path ./target/debug

# Test specific chapter
mdbook test path/to/chapter.md
```

### Cleaning

```bash
# Clean build directory
mdbook clean

# Remove specific build
rm -rf book/
```

## Markdown Extensions

### Code Blocks

````markdown
```rust
fn main() {
    println!("Hello, world!");
}
```

```rust,editable
// This code can be edited in browser
fn main() {
    println!("Try editing me!");
}
```

```rust,ignore
// This code won't be tested
fn incomplete() {
```

```rust,no_run
// Compiles but doesn't run during tests
fn main() {
    std::process::exit(1);
}
```

```rust,should_panic
// Expected to panic
fn main() {
    panic!("Expected panic");
}
```

```python
def greet(name):
    print(f"Hello, {name}!")
```

```bash
#!/bin/bash
echo "Hello from bash"
```
````

### Include Files

```markdown
<!-- Include entire file -->
{{#include path/to/file.rs}}

<!-- Include specific lines -->
{{#include path/to/file.rs:10:20}}

<!-- Include from line to end -->
{{#include path/to/file.rs:10:}}

<!-- Include with anchor -->
{{#include path/to/file.rs:my_anchor}}
```

### Rust Playground

````markdown
```rust,editable
{{#playpen example.rs}}
```

```rust
{{#rustdoc_include path/to/lib.rs}}
```
````

## Customization

### Custom CSS

```css
/* custom.css */
:root {
    --sidebar-width: 300px;
    --page-padding: 20px;
    --content-max-width: 900px;
}

.content {
    font-size: 18px;
    line-height: 1.8;
}

.chapter {
    padding: 2em;
}

code {
    font-family: 'Fira Code', monospace;
}

pre {
    border-radius: 8px;
}
```

### Custom JavaScript

```javascript
// custom.js
window.addEventListener('load', function() {
    // Add custom functionality
    console.log('Book loaded');

    // Add copy button to code blocks
    document.querySelectorAll('pre > code').forEach(function(code) {
        const button = document.createElement('button');
        button.textContent = 'Copy';
        button.onclick = function() {
            navigator.clipboard.writeText(code.textContent);
            button.textContent = 'Copied!';
            setTimeout(() => button.textContent = 'Copy', 2000);
        };
        code.parentElement.insertBefore(button, code);
    });
});
```

### Custom Theme

```toml
# book.toml
[output.html]
theme = "my-theme"

# Create theme directory
# mkdir -p my-theme
# Copy and modify default theme files
```

```bash
# Extract default theme
mdbook init --theme

# Files created in theme/:
# - index.hbs        # Main template
# - head.hbs         # HTML head
# - header.hbs       # Page header
# - chrome.css       # UI styles
# - general.css      # Content styles
# - variables.css    # CSS variables
```

## Preprocessors

### Built-in Preprocessors

```toml
# Enable links preprocessor
[preprocessor.links]

# Example usage in Markdown:
# [Rust](https://www.rust-lang.org/)
```

### Custom Preprocessor

```rust
// my-preprocessor/src/main.rs
use mdbook::preprocess::{Preprocessor, PreprocessorContext};
use mdbook::book::Book;
use std::io;

struct MyPreprocessor;

impl Preprocessor for MyPreprocessor {
    fn name(&self) -> &str {
        "my-preprocessor"
    }

    fn run(&self, ctx: &PreprocessorContext, mut book: Book) -> Result<Book, Error> {
        // Process book content
        Ok(book)
    }
}

fn main() {
    let preprocessor = MyPreprocessor;
    if let Err(e) = mdbook::preprocess::handle_preprocessing(&preprocessor) {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}
```

```toml
# book.toml
[preprocessor.my-preprocessor]
command = "my-preprocessor"
```

## Deployment

### GitHub Pages

```bash
# Build book
mdbook build

# Initialize git (if needed)
git init
git add .
git commit -m "Initial commit"

# Create gh-pages branch
git checkout --orphan gh-pages
git reset --hard
cp -r book/* .
rm -rf book src
git add .
git commit -m "Deploy book"
git push origin gh-pages

# Or use GitHub Actions
```

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy mdBook

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup mdBook
        uses: peaceiris/actions-mdbook@v1
        with:
          mdbook-version: 'latest'

      - name: Build
        run: mdbook build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./book
```

### Netlify

```toml
# netlify.toml
[build]
command = "mdbook build"
publish = "book"

[build.environment]
RUST_VERSION = "1.70.0"
```

### Docker

```dockerfile
# Dockerfile
FROM rust:1.70 as builder

RUN cargo install mdbook

WORKDIR /book
COPY . .
RUN mdbook build

FROM nginx:alpine
COPY --from=builder /book/book /usr/share/nginx/html
```

## Common Patterns

### Multi-Language Book

```toml
# book.toml
[book]
multilingual = true

[output.html]
redirect = { "/" = "/en/" }

# Directory structure:
# src/
# ├── en/
# │   ├── SUMMARY.md
# │   └── chapter_1.md
# └── es/
#     ├── SUMMARY.md
#     └── chapter_1.md
```

### Code Examples Project

```markdown
<!-- Link to example project -->
See the [full example](https://github.com/user/repo/tree/main/examples/basic)

<!-- Include code from example -->
{{#include ../../examples/basic/src/main.rs}}
```

### Versioned Documentation

```bash
#!/bin/bash
# build_versions.sh

VERSIONS=("v1.0" "v1.1" "v2.0")

for version in "${VERSIONS[@]}"; do
    git checkout $version
    mdbook build -d "book/$version"
done

# Create index.html for version selection
```

## Best Practices

### Content Organization

```markdown
# Recommended structure:
src/
├── SUMMARY.md
├── introduction.md
├── guide/
│   ├── README.md         # Chapter intro
│   ├── basics.md
│   └── advanced.md
├── reference/
│   ├── README.md
│   ├── api.md
│   └── cli.md
├── examples/
│   └── tutorial.md
└── appendix/
    ├── glossary.md
    └── resources.md
```

### Markdown Style

```markdown
# Use consistent heading levels

## Chapter Title

### Section

#### Subsection

# Use relative links
[Link to other chapter](../other/chapter.md)

# Use descriptive alt text for images
![Architecture diagram showing components](./images/arch.png)

# Include language in code blocks
```rust
fn main() {}
```

# Use admonitions (with appropriate CSS)
> **Note**: Important information

> **Warning**: Be careful here
```

### Performance Tips

```bash
# Minimize preprocessors
# Use relative links
# Optimize images
# Enable search caching

[output.html.search]
limit-results = 20
```

## Troubleshooting

```bash
# Build fails
mdbook build -v          # Verbose output

# Links not working
# Use relative links: ./file.md or ../other/file.md

# Search not working
[output.html.search]
enable = true

# Changes not reflecting
mdbook clean && mdbook build

# Port already in use
mdbook serve -p 3001

# Code not highlighting
# Ensure language is specified in code blocks
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `mdbook init` | Create new book |
| `mdbook build` | Build book |
| `mdbook serve` | Serve with live reload |
| `mdbook test` | Test code examples |
| `mdbook clean` | Clean build directory |
| `mdbook watch` | Watch for changes |

mdBook is an excellent tool for creating beautiful, fast, and maintainable documentation, perfect for technical books, tutorials, API documentation, and user guides.
