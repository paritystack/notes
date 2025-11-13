# ctags

ctags is a tool that generates an index (or "tag") file of names found in source and header files, enabling efficient code navigation in text editors. It supports numerous programming languages and integrates seamlessly with Vim, Emacs, and other editors.

## Overview

ctags creates a database of language objects (functions, classes, variables, etc.) found in source files, allowing editors to quickly jump to definitions. Modern implementations include Exuberant Ctags and Universal Ctags.

**Key Features:**
- Multi-language support (C, C++, Python, Java, JavaScript, etc.)
- Editor integration (Vim, Emacs, Sublime, VS Code)
- Recursive directory scanning
- Custom tag patterns
- Symbol cross-referencing
- Incremental updates

## Installation

```bash
# Ubuntu/Debian - Universal Ctags (recommended)
sudo apt update
sudo apt install universal-ctags

# Or Exuberant Ctags (older)
sudo apt install exuberant-ctags

# macOS - Universal Ctags
brew install --HEAD universal-ctags/universal-ctags/universal-ctags

# CentOS/RHEL
sudo yum install ctags

# Arch Linux
sudo pacman -S ctags

# From source (Universal Ctags)
git clone https://github.com/universal-ctags/ctags.git
cd ctags
./autogen.sh
./configure
make
sudo make install

# Verify installation
ctags --version
```

## Basic Usage

### Generating Tags

```bash
# Generate tags for current directory
ctags *

# Recursive tag generation
ctags -R

# Specific files
ctags file1.c file2.c file3.h

# Multiple languages
ctags -R src/ include/

# Generate tags for specific language
ctags -R --languages=C,C++

# Exclude languages
ctags -R --languages=-JavaScript,-HTML

# Follow symbolic links
ctags -R --links=yes
```

### Tag File Options

```bash
# Specify output file
ctags -o mytags -R

# Append to existing tags
ctags -a -R new_directory/

# Create tag file with extra information
ctags -R --fields=+iaS --extras=+q

# Sort tags file
ctags -R --sort=yes

# Case-insensitive sorting
ctags -R --sort=foldcase
```

## Vim Integration

### Basic Configuration

```vim
" Add to ~/.vimrc
set tags=./tags,tags;$HOME

" Search for tags file in current directory and up to $HOME
set tags=./tags;/
```

### Vim Commands

```vim
" Jump to definition
Ctrl+]          " Jump to tag under cursor
g Ctrl+]        " Show list if multiple matches

" Return from jump
Ctrl+T          " Jump back (pop tag stack)
Ctrl+O          " Jump to previous location

" Navigation
:tag function   " Jump to tag
:ts pattern     " List matching tags
:tn             " Next matching tag
:tp             " Previous matching tag

" Tag stack
:tags           " Show tag stack
:pop            " Pop from tag stack

" Split window navigation
Ctrl+W ]        " Split window and jump to tag
Ctrl+W g ]      " Split and list matches
```

### Advanced Vim Configuration

```vim
" ~/.vimrc
" Set tags file locations
set tags=./tags,tags;$HOME

" Enable tag stack
set tagstack

" Show tag preview in popup
set completeopt=menuone,preview

" Custom key mappings
nnoremap <C-]> g<C-]>       " Always show list if multiple matches
nnoremap <leader>t :tag<Space>
nnoremap <leader>] :tselect<CR>
nnoremap <leader>[ :pop<CR>

" Split navigation
nnoremap <C-\> :tab split<CR>:exec("tag ".expand("<cword>"))<CR>
nnoremap <A-]> :vsp <CR>:exec("tag ".expand("<cword>"))<CR>

" Auto-regenerate tags
autocmd BufWritePost *.c,*.cpp,*.h,*.py silent! !ctags -R &
```

### Vim with Tagbar Plugin

```vim
" Install with vim-plug
Plug 'majutsushi/tagbar'

" Configuration
nmap <F8> :TagbarToggle<CR>

let g:tagbar_width = 30
let g:tagbar_autofocus = 1
let g:tagbar_sort = 0

" Custom language configuration
let g:tagbar_type_go = {
    \ 'ctagstype' : 'go',
    \ 'kinds'     : [
        \ 'p:package',
        \ 'i:imports',
        \ 'c:constants',
        \ 'v:variables',
        \ 't:types',
        \ 'n:interfaces',
        \ 'w:fields',
        \ 'e:embedded',
        \ 'm:methods',
        \ 'r:constructor',
        \ 'f:functions'
    \ ],
    \ 'sro' : '.',
    \ 'kind2scope' : {
        \ 't' : 'ctype',
        \ 'n' : 'ntype'
    \ },
    \ 'scope2kind' : {
        \ 'ctype' : 't',
        \ 'ntype' : 'n'
    \ },
\ }
```

## Language-Specific Features

### C/C++

```bash
# C/C++ with all features
ctags -R \
    --c-kinds=+p \
    --c++-kinds=+p \
    --fields=+iaS \
    --extras=+q

# Include system headers
ctags -R --c-kinds=+px --fields=+iaS --extras=+q \
    /usr/include \
    /usr/local/include \
    .

# Kernel-style projects
ctags -R \
    --exclude=.git \
    --exclude=build \
    --exclude=Documentation \
    --languages=C \
    --langmap=c:.c.h \
    --c-kinds=+px \
    --fields=+iaS \
    --extras=+q
```

### Python

```bash
# Python projects
ctags -R \
    --languages=Python \
    --python-kinds=-i \
    --fields=+l

# Include virtualenv
ctags -R \
    --languages=Python \
    --fields=+l \
    . \
    venv/lib/python*/site-packages/
```

### JavaScript/TypeScript

```bash
# JavaScript
ctags -R \
    --languages=JavaScript \
    --exclude=node_modules \
    --exclude=dist \
    --exclude=build

# TypeScript
ctags -R \
    --languages=TypeScript \
    --exclude=node_modules \
    --exclude=*.min.js
```

### Java

```bash
# Java projects
ctags -R \
    --languages=Java \
    --exclude=.git \
    --exclude=target \
    --exclude=*.class

# Include JAR dependencies (if unpacked)
ctags -R src/ lib/
```

## Advanced Usage

### Custom Configuration

```bash
# ~/.ctags.d/local.ctags (Universal Ctags)
--recurse=yes
--tag-relative=yes
--exclude=.git
--exclude=.svn
--exclude=.hg
--exclude=node_modules
--exclude=bower_components
--exclude=*.min.js
--exclude=*.swp
--exclude=*.bak
--exclude=*.pyc
--exclude=*.class
--exclude=target
--exclude=build
--exclude=dist

# Language-specific
--langdef=markdown
--langmap=markdown:.md.markdown.mdown.mkd.mkdn
--regex-markdown=/^#{1}[ \t]+(.+)/. \1/h,heading1/
--regex-markdown=/^#{2}[ \t]+(.+)/..  \1/h,heading2/
--regex-markdown=/^#{3}[ \t]+(.+)/...   \1/h,heading3/
```

### Project-Specific Tags

```bash
# .git/hooks/post-commit
#!/bin/bash
ctags -R &

# Make executable
chmod +x .git/hooks/post-commit

# Or use Makefile
.PHONY: tags
tags:
	ctags -R --fields=+iaS --extras=+q

.PHONY: tags-clean
tags-clean:
	rm -f tags
```

### Filtering and Exclusions

```bash
# Exclude directories
ctags -R --exclude=build --exclude=.git --exclude=node_modules

# Exclude files by pattern
ctags -R --exclude=*.min.js --exclude=*.test.js

# Include only specific directories
ctags -R src/ include/

# Custom exclusions file
echo "build/" > .ctagsignore
echo "*.min.js" >> .ctagsignore
ctags -R --exclude=@.ctagsignore
```

## Scripting with ctags

### Automated Tag Generation

```bash
#!/bin/bash
# update_tags.sh

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$PROJECT_ROOT"

echo "Generating tags for: $PROJECT_ROOT"

ctags -R \
    --fields=+iaS \
    --extras=+q \
    --exclude=.git \
    --exclude=build \
    --exclude=node_modules \
    --exclude=*.min.js

echo "Tags file generated: $PROJECT_ROOT/tags"
```

### Multi-Project Tags

```bash
#!/bin/bash
# generate_all_tags.sh

PROJECTS=(
    "$HOME/projects/project1"
    "$HOME/projects/project2"
    "$HOME/projects/lib/common"
)

for project in "${PROJECTS[@]}"; do
    if [ -d "$project" ]; then
        echo "Generating tags for $project"
        (cd "$project" && ctags -R)
    fi
done

# Merge tags files
cat ~/projects/*/tags | sort -u > ~/projects/all_tags
```

### Find Symbol Across Projects

```bash
#!/bin/bash
# find_symbol.sh

SYMBOL=$1

if [ -z "$SYMBOL" ]; then
    echo "Usage: $0 <symbol>"
    exit 1
fi

# Search in tags file
echo "Searching for: $SYMBOL"
echo ""

grep "^$SYMBOL" tags | while IFS=$'\t' read tag file pattern rest; do
    echo "File: $file"
    echo "Pattern: $pattern"
    echo "---"
done
```

## Integration with Other Editors

### Emacs

```elisp
;; Add to ~/.emacs or ~/.emacs.d/init.el

;; Enable etags (similar to ctags)
(setq tags-table-list '("./TAGS" "../TAGS" "../../TAGS"))

;; Key bindings
(global-set-key (kbd "M-.") 'find-tag)
(global-set-key (kbd "M-*") 'pop-tag-mark)
(global-set-key (kbd "M-,") 'tags-loop-continue)

;; Generate tags for project
(defun my-generate-tags ()
  (interactive)
  (shell-command "ctags -e -R ."))

(global-set-key (kbd "C-c g") 'my-generate-tags)
```

### VS Code

```json
// settings.json
{
    "ctagsFile": "tags",
    "ctagsPath": "/usr/bin/ctags"
}

// Install extension
// ext install jaydenlin.ctags-support
```

### Sublime Text

```json
// Settings - User
{
    "tags_path": "tags",
    "ctags_command": "/usr/bin/ctags -R --fields=+iaS --extras=+q"
}

// Install CTags package via Package Control
```

## Common Patterns

### Monorepo Tag Management

```bash
#!/bin/bash
# monorepo_tags.sh

# Root tags
ctags -R --fields=+iaS --extras=+q -o tags.root .

# Per-service tags
for service in services/*; do
    if [ -d "$service" ]; then
        (cd "$service" && ctags -R -o tags .)
    fi
done

# Merge all tags
find . -name "tags" -exec cat {} \; | sort -u > tags
```

### Language-Specific Tag Files

```bash
#!/bin/bash
# Generate separate tags for each language

# C/C++ tags
ctags -R -o tags.c --languages=C,C++ .

# Python tags
ctags -R -o tags.py --languages=Python .

# JavaScript tags
ctags -R -o tags.js --languages=JavaScript --exclude=node_modules .

# Merge all
cat tags.* | sort -u > tags
```

### Incremental Updates

```bash
#!/bin/bash
# update_changed.sh

# Get changed files since last tag generation
CHANGED=$(find . -type f -newer tags \( -name "*.c" -o -name "*.h" \))

if [ ! -z "$CHANGED" ]; then
    echo "Updating tags for changed files"

    # Generate tags for changed files
    ctags -a $CHANGED

    # Sort tags file
    sort -u tags -o tags
fi
```

## Best Practices

### Recommended Configuration

```bash
# ~/.ctags or ~/.ctags.d/default.ctags (Universal Ctags)

# Recurse by default
--recurse=yes

# Tag relative paths
--tag-relative=yes

# Additional fields
--fields=+iaS
--extras=+q

# Common exclusions
--exclude=.git
--exclude=.svn
--exclude=node_modules
--exclude=bower_components
--exclude=*.min.js
--exclude=*.min.css
--exclude=*.map
--exclude=build
--exclude=dist
--exclude=target
--exclude=*.pyc
--exclude=*.class
--exclude=.DS_Store

# Sort tags
--sort=yes

# Language-specific
--languages=all
--c-kinds=+px
--c++-kinds=+px
--python-kinds=-i
```

### Git Integration

```bash
# .gitignore
tags
tags.lock
tags.temp
TAGS

# .git/hooks/post-checkout
#!/bin/bash
ctags -R &

# .git/hooks/post-merge
#!/bin/bash
ctags -R &
```

### Performance Tips

```bash
# Use parallel processing for large projects
find . -name "*.c" -o -name "*.h" | xargs -P 4 -n 50 ctags -a

# Generate tags in background
ctags -R &

# Use faster sorting
ctags -R --sort=no
LC_ALL=C sort tags -o tags

# Exclude large dependency directories
ctags -R --exclude=vendor --exclude=node_modules
```

## Troubleshooting

```bash
# Tags file not found in Vim
:set tags?          # Check tags path
:set tags=./tags;/  # Set tags path

# Duplicate entries
sort -u tags -o tags

# Wrong language detected
ctags --list-languages        # Show supported languages
ctags --list-maps            # Show file extensions
ctags -R --languages=C,C++   # Force specific languages

# Performance issues
ctags -R --exclude=node_modules --exclude=vendor

# Tags not updating
rm tags
ctags -R

# Vim not jumping to correct location
# Regenerate with line numbers
ctags -R --fields=+n

# Check tag format
head -n 20 tags
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `ctags -R` | Generate tags recursively |
| `ctags -a` | Append to tags |
| `ctags --list-languages` | Show supported languages |
| `Ctrl+]` | Vim: Jump to tag |
| `Ctrl+T` | Vim: Return from tag |
| `:ts` | Vim: List tags |
| `:tag name` | Vim: Jump to tag |
| `--exclude=DIR` | Exclude directory |
| `--languages=LANG` | Specific languages |
| `--fields=+iaS` | Extra tag fields |

ctags is an essential tool for code navigation, enabling developers to efficiently explore and understand large codebases by providing instant access to symbol definitions and references.
