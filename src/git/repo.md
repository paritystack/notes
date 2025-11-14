# Git Repository Management

A comprehensive guide to Git repository operations, patterns, and best practices for effective repository management.

## Overview

A Git repository is a data structure that stores metadata and object database for a project's version history. Understanding repository management is crucial for efficient collaboration, maintenance, and scaling of software projects.

**Key Concepts:**
- **Repository**: Complete project history and metadata
- **Working Tree**: Current checkout of project files
- **Index (Staging Area)**: Preparation area for commits
- **Bare Repository**: Repository without working tree
- **Clone**: Local copy of a repository
- **Remote**: Reference to another repository

## Repository Anatomy

### Repository Structure

```
.git/
├── HEAD               # Points to current branch
├── config             # Repository-specific configuration
├── description        # Repository description (for GitWeb)
├── hooks/             # Client and server-side hook scripts
├── info/              # Global exclude patterns
│   └── exclude
├── objects/           # All Git objects (commits, trees, blobs)
│   ├── info/
│   └── pack/          # Packed objects for efficiency
├── refs/              # References to commits
│   ├── heads/         # Local branches
│   ├── remotes/       # Remote branches
│   └── tags/          # Tags
├── logs/              # Reference logs
├── index              # Staging area (binary)
└── packed-refs        # Packed references for efficiency
```

### Understanding Key Components

```bash
# View current HEAD
cat .git/HEAD

# List all references
git show-ref

# View repository config
cat .git/config

# Inspect staging area
git ls-files --stage

# View object database
find .git/objects -type f
```

## Repository Initialization

### Creating a New Repository

```bash
# Initialize new repository
git init
git init project-name
git init --bare repo.git  # Bare repository (no working tree)

# Initialize with specific branch name
git init --initial-branch=main
git init -b main

# Initialize with template
git init --template=/path/to/template

# Reinitialize existing repository (safe operation)
git init
```

### Repository Configuration

```bash
# Local (repository-specific)
git config user.name "John Doe"
git config user.email "john@example.com"

# Global (user-level)
git config --global core.editor "vim"
git config --global init.defaultBranch main

# System (all users)
git config --system core.compression 9

# View configuration hierarchy
git config --list --show-origin

# Edit config file directly
git config --edit
git config --global --edit
git config --system --edit
```

### Essential Repository Settings

```bash
# Set file permissions
git config core.fileMode true

# Handle line endings
git config core.autocrlf input     # Linux/macOS
git config core.autocrlf true      # Windows

# Ignore file permissions
git config core.fileMode false

# Set default push behavior
git config push.default simple

# Enable rerere (reuse recorded resolution)
git config rerere.enabled true

# Set default pull strategy
git config pull.rebase false   # Merge (default)
git config pull.rebase true    # Rebase
git config pull.ff only        # Fast-forward only
```

## Cloning Repositories

### Basic Cloning

```bash
# Clone via HTTPS
git clone https://github.com/user/repo.git
git clone https://github.com/user/repo.git myproject

# Clone via SSH
git clone git@github.com:user/repo.git

# Clone specific branch
git clone -b develop https://github.com/user/repo.git

# Shallow clone (limited history)
git clone --depth 1 https://github.com/user/repo.git
git clone --shallow-since=2023-01-01 https://github.com/user/repo.git

# Clone single branch
git clone --single-branch --branch main https://github.com/user/repo.git
```

### Advanced Cloning Options

```bash
# Clone bare repository
git clone --bare https://github.com/user/repo.git repo.git

# Clone mirror (complete copy including all refs)
git clone --mirror https://github.com/user/repo.git

# Clone with submodules
git clone --recursive https://github.com/user/repo.git
git clone --recurse-submodules https://github.com/user/repo.git

# Partial clone (lazy fetch)
git clone --filter=blob:none https://github.com/user/repo.git
git clone --filter=tree:0 https://github.com/user/repo.git

# Clone with sparse checkout
git clone --sparse https://github.com/user/repo.git
cd repo
git sparse-checkout init --cone
git sparse-checkout set src/ docs/
```

### Shallow Clone Operations

```bash
# Deepen shallow clone
git fetch --deepen=100
git fetch --unshallow  # Fetch complete history

# Fetch specific branch in shallow repo
git remote set-branches origin 'feature/*'
git fetch --depth 1 origin feature/new

# Prune old history
git fetch --depth=1
```

## Remote Repository Management

### Working with Remotes

```bash
# Add remote
git remote add origin https://github.com/user/repo.git
git remote add upstream https://github.com/original/repo.git

# List remotes
git remote -v
git remote show origin

# Rename remote
git remote rename origin upstream

# Change remote URL
git remote set-url origin git@github.com:user/repo.git
git remote set-url --add origin https://gitlab.com/user/repo.git

# Remove remote
git remote remove origin
git remote rm origin
```

### Remote Tracking

```bash
# View remote branches
git branch -r
git branch -a  # All branches (local + remote)

# Track remote branch
git checkout -b feature origin/feature
git checkout --track origin/feature
git branch -u origin/feature  # Set upstream for current branch

# View tracking relationships
git branch -vv

# Fetch from remote
git fetch origin
git fetch --all  # Fetch from all remotes
git fetch --prune  # Remove deleted remote branches

# Pull changes
git pull origin main
git pull --rebase origin main

# Push to remote
git push origin main
git push -u origin feature  # Set upstream and push
git push --all origin  # Push all branches
git push --tags  # Push tags
git push --force-with-lease  # Safer force push
```

### Multiple Remotes

```bash
# Setup multiple remotes for push
git remote set-url --add --push origin git@github.com:user/repo.git
git remote set-url --add --push origin git@gitlab.com:user/repo.git

# Verify configuration
git remote -v

# Push to specific remote
git push github main
git push gitlab main

# Pull from specific remote
git pull upstream main
```

### Remote Branches Management

```bash
# Delete remote branch
git push origin --delete feature-branch
git push origin :feature-branch  # Older syntax

# Clean up stale remote references
git remote prune origin
git fetch --prune

# Update remote branch
git push origin main:main
git push origin local-branch:remote-branch

# Push all tags
git push origin --tags

# Delete remote tag
git push origin --delete tag-name
git push origin :refs/tags/tag-name
```

## Repository Patterns

### Monorepo Pattern

A single repository containing multiple projects or services.

**Advantages:**
- Unified versioning
- Atomic commits across projects
- Simplified dependency management
- Easier code sharing and refactoring

**Structure:**
```
monorepo/
├── services/
│   ├── api/
│   ├── web/
│   └── mobile/
├── packages/
│   ├── shared-ui/
│   └── utils/
├── tools/
└── docs/
```

**Best Practices:**
```bash
# Use sparse checkout for large monorepos
git sparse-checkout init --cone
git sparse-checkout set services/api packages/shared-ui

# Use shallow clone for CI/CD
git clone --depth 1 --filter=blob:none https://repo.git

# Tag strategy for monorepos
git tag api-v1.2.0
git tag web-v2.0.0
git tag shared-ui-v0.5.0
```

### Multi-Repo Pattern

Separate repositories for each project or service.

**Advantages:**
- Independent versioning
- Smaller repositories
- Clearer access control
- Service isolation

**Management:**
```bash
# Use git submodules
git submodule add https://github.com/user/lib.git libs/lib

# Use meta-repositories
git clone https://github.com/org/meta-repo.git
cd meta-repo
./scripts/clone-all.sh

# Use repo tool (Android-style)
repo init -u https://github.com/org/manifest.git
repo sync
```

### Fork Workflow Pattern

Standard open-source contribution model.

```bash
# 1. Fork repository on GitHub

# 2. Clone your fork
git clone git@github.com:yourname/repo.git
cd repo

# 3. Add upstream remote
git remote add upstream https://github.com/original/repo.git

# 4. Create feature branch
git checkout -b feature/new-feature

# 5. Keep fork updated
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# 6. Work on feature
git add .
git commit -m "Add new feature"
git push origin feature/new-feature

# 7. Create pull request on GitHub

# 8. Update PR with upstream changes
git fetch upstream
git rebase upstream/main
git push --force-with-lease origin feature/new-feature
```

### Gitflow Pattern

Structured branching model for releases.

```bash
# Branch structure
main          # Production code
develop       # Integration branch
feature/*     # Feature branches
release/*     # Release preparation
hotfix/*      # Production fixes

# Start feature
git checkout -b feature/user-auth develop

# Finish feature
git checkout develop
git merge --no-ff feature/user-auth
git branch -d feature/user-auth
git push origin develop

# Start release
git checkout -b release/1.0.0 develop
# Bump version, update changelog
git commit -m "Prepare release 1.0.0"

# Finish release
git checkout main
git merge --no-ff release/1.0.0
git tag -a v1.0.0 -m "Release 1.0.0"
git checkout develop
git merge --no-ff release/1.0.0
git branch -d release/1.0.0

# Hotfix
git checkout -b hotfix/1.0.1 main
# Fix critical bug
git commit -m "Fix critical bug"
git checkout main
git merge --no-ff hotfix/1.0.1
git tag -a v1.0.1
git checkout develop
git merge --no-ff hotfix/1.0.1
git branch -d hotfix/1.0.1
```

### Trunk-Based Development

Single main branch with short-lived feature branches.

```bash
# Main workflow
git checkout main
git pull origin main
git checkout -b feature/quick-fix
# Make changes
git commit -m "Quick fix"
git push origin feature/quick-fix
# Create PR, review, merge quickly

# Feature flags for incomplete features
if (featureFlags.newUI) {
  // New UI code
} else {
  // Old UI code
}

# Continuous integration
# All commits must pass tests before merge
```

## Submodules

### Working with Submodules

```bash
# Add submodule
git submodule add https://github.com/user/lib.git libs/lib
git submodule add -b main https://github.com/user/lib.git libs/lib

# Clone repository with submodules
git clone --recursive https://github.com/user/repo.git

# Initialize submodules after clone
git submodule init
git submodule update

# Update submodules
git submodule update --remote
git submodule update --remote --merge

# Update specific submodule
git submodule update --remote libs/lib

# Execute command in all submodules
git submodule foreach git pull origin main
git submodule foreach 'git checkout main && git pull'
```

### Advanced Submodule Operations

```bash
# Remove submodule
git submodule deinit libs/lib
git rm libs/lib
rm -rf .git/modules/libs/lib

# Change submodule URL
git config submodule.libs/lib.url https://new-url.git
git submodule sync libs/lib
git submodule update --remote libs/lib

# Pin submodule to specific commit
cd libs/lib
git checkout abc1234
cd ../..
git add libs/lib
git commit -m "Pin lib to specific version"

# View submodule status
git submodule status
git submodule summary

# Recursive submodule operations
git clone --recursive --shallow-submodules https://github.com/user/repo.git
git submodule update --init --recursive
```

### Submodule Configuration

```bash
# .gitmodules file
[submodule "libs/lib"]
    path = libs/lib
    url = https://github.com/user/lib.git
    branch = main
    update = merge

# Ignore changes in submodule
git config submodule.libs/lib.ignore dirty
git config submodule.libs/lib.ignore untracked
git config submodule.libs/lib.ignore all

# Set default update strategy
git config submodule.recurse true
git config submodule.libs/lib.update merge
```

## Subtrees

### Using Subtrees

```bash
# Add subtree
git subtree add --prefix=lib https://github.com/user/lib.git main --squash

# Pull updates from subtree
git subtree pull --prefix=lib https://github.com/user/lib.git main --squash

# Push changes to subtree
git subtree push --prefix=lib https://github.com/user/lib.git main

# Split subtree into separate branch
git subtree split --prefix=lib -b lib-only

# Add subtree from existing code
git remote add lib-origin https://github.com/user/lib.git
git subtree add --prefix=lib lib-origin main
```

### Subtree vs Submodule

**Submodules:**
- Separate repositories
- Track specific commits
- Require initialization
- Better for independent projects
- Smaller parent repo size

**Subtrees:**
- Merged into parent repository
- No special commands needed for cloning
- Simplified workflow
- Better for vendoring
- Larger parent repo size

```bash
# Migrate from submodule to subtree
git submodule deinit libs/lib
git rm libs/lib
rm -rf .git/modules/libs/lib
git subtree add --prefix=libs/lib https://github.com/user/lib.git main --squash
```

## Repository Maintenance

### Optimization

```bash
# Garbage collection
git gc
git gc --aggressive  # More thorough, slower

# Prune unreachable objects
git prune
git prune --dry-run  # See what would be deleted

# Clean up repository
git clean -n   # Dry run
git clean -fd  # Remove untracked files and directories
git clean -fdx # Include ignored files

# Repack objects
git repack
git repack -a -d -f --depth=250 --window=250

# Optimize repository
git gc --aggressive --prune=now

# Reduce repository size
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Repository Verification

```bash
# Verify repository integrity
git fsck
git fsck --full
git fsck --unreachable

# Check object database
git count-objects -v
git count-objects -vH  # Human-readable

# Verify connectivity
git fsck --connectivity-only

# Find corrupted objects
git fsck --lost-found
```

### Repository Statistics

```bash
# Repository size
du -sh .git/

# Object count and size
git count-objects -v

# Largest objects
git rev-list --objects --all |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  sed -n 's/^blob //p' |
  sort -rnk2 |
  head -20

# Commit count by author
git shortlog -sn

# Repository activity
git log --all --oneline --graph --decorate

# File change frequency
git log --all --format=format: --name-only | sort | uniq -c | sort -rn | head -20
```

### Backup and Recovery

```bash
# Create backup
git bundle create repo-backup.bundle --all
git clone repo-backup.bundle repo-restored

# Incremental backup
git bundle create repo-incremental.bundle main ^backup-branch
git bundle verify repo-backup.bundle

# Mirror repository
git clone --mirror https://github.com/user/repo.git backup-repo.git
cd backup-repo.git
git remote update

# Export repository
git fast-export --all > repo-export.txt
git fast-import < repo-export.txt

# Recover deleted branch
git reflog
git checkout -b recovered-branch abc1234

# Recover deleted commits
git fsck --lost-found
git show <dangling-commit-hash>
git cherry-pick <hash>
```

## Repository Templates

### Creating Templates

```bash
# Create template directory
mkdir -p ~/.git-templates/hooks

# Add template files
cat > ~/.git-templates/hooks/pre-commit << 'EOF'
#!/bin/bash
# Run linter before commit
npm run lint
EOF
chmod +x ~/.git-templates/hooks/pre-commit

# Configure Git to use template
git config --global init.templateDir ~/.git-templates

# Initialize with template
git init  # Uses global template
git init --template=/path/to/template
```

### Template Structure

```
.git-templates/
├── hooks/
│   ├── pre-commit
│   ├── commit-msg
│   ├── pre-push
│   └── post-merge
├── info/
│   └── exclude
└── description
```

### Common Template Files

**`.git-templates/info/exclude`:**
```
*.swp
*.swo
*~
.DS_Store
.idea/
.vscode/
```

**`.git-templates/hooks/commit-msg`:**
```bash
#!/bin/bash
# Enforce commit message format
commit_msg=$(cat "$1")
pattern="^(feat|fix|docs|style|refactor|test|chore):"

if ! echo "$commit_msg" | grep -qE "$pattern"; then
    echo "Error: Commit message must start with type (feat|fix|docs|...)"
    exit 1
fi
```

## Migration and Conversion

### Migrating from Other VCS

**From SVN:**
```bash
# Clone SVN repository
git svn clone https://svn.example.com/repo --stdlayout

# Or with custom layout
git svn clone https://svn.example.com/repo \
  --trunk=main \
  --branches=branches \
  --tags=tags

# Convert authors
cat > authors.txt << EOF
svnuser = Git User <user@example.com>
EOF
git svn clone --authors-file=authors.txt https://svn.example.com/repo

# Clean up
git remote add origin https://github.com/user/repo.git
git push -u origin main
```

**From Mercurial:**
```bash
# Use hg-git plugin
hg bookmark -r default main
git clone hg::https://hg.example.com/repo

# Or use fast-export
git init repo
cd repo
hg-fast-export.sh -r /path/to/hg/repo
git remote add origin https://github.com/user/repo.git
git push -u origin main
```

### Repository Splitting

```bash
# Split subdirectory into new repo
git filter-branch --subdirectory-filter subdir -- --all

# Or use filter-repo (faster, recommended)
git filter-repo --path subdir/

# Create new repository
git remote add origin https://github.com/user/new-repo.git
git push -u origin main

# Split by path patterns
git filter-repo --path src/module1/ --path docs/module1/
```

### Repository Merging

```bash
# Merge multiple repos into one
git remote add repo1 https://github.com/user/repo1.git
git fetch repo1
git merge repo1/main --allow-unrelated-histories

# Move to subdirectory
git filter-repo --to-subdirectory-filter repo1/

# Repeat for other repositories
git remote add repo2 https://github.com/user/repo2.git
git fetch repo2
git filter-repo --to-subdirectory-filter repo2/
git merge repo2/main --allow-unrelated-histories
```

### History Rewriting

```bash
# Remove sensitive data
git filter-branch --tree-filter 'rm -f passwords.txt' HEAD
# Better: use filter-repo
git filter-repo --invert-paths --path passwords.txt

# Change author information
git filter-branch --env-filter '
if [ "$GIT_COMMITTER_EMAIL" = "old@example.com" ]; then
    export GIT_COMMITTER_NAME="New Name"
    export GIT_COMMITTER_EMAIL="new@example.com"
fi
if [ "$GIT_AUTHOR_EMAIL" = "old@example.com" ]; then
    export GIT_AUTHOR_NAME="New Name"
    export GIT_AUTHOR_EMAIL="new@example.com"
fi
' --tag-name-filter cat -- --branches --tags

# Remove large files
git filter-repo --strip-blobs-bigger-than 10M

# Simplify history
git filter-branch --commit-filter '
    if [ "$GIT_AUTHOR_NAME" = "Temporary User" ]; then
        skip_commit "$@"
    else
        git commit-tree "$@"
    fi
' HEAD
```

## Advanced Repository Operations

### Worktrees

Work on multiple branches simultaneously.

```bash
# Add worktree
git worktree add ../project-feature1 feature1
git worktree add -b feature2 ../project-feature2

# List worktrees
git worktree list

# Remove worktree
git worktree remove ../project-feature1
git worktree prune

# Lock/unlock worktree
git worktree lock ../project-feature1
git worktree unlock ../project-feature1

# Move worktree
git worktree move ../project-feature1 ../new-location
```

### Sparse Checkout

Checkout only specific directories.

```bash
# Enable sparse checkout
git sparse-checkout init
git sparse-checkout init --cone  # Cone mode (recommended)

# Set patterns
git sparse-checkout set src/ docs/
git sparse-checkout add tests/

# View patterns
git sparse-checkout list

# Disable sparse checkout
git sparse-checkout disable
```

### Partial Clone

Clone without all objects (lazy loading).

```bash
# Blob-less clone
git clone --filter=blob:none https://github.com/user/huge-repo.git

# Tree-less clone
git clone --filter=tree:0 https://github.com/user/huge-repo.git

# Combine with shallow clone
git clone --depth=1 --filter=blob:none https://github.com/user/repo.git

# Fetch missing objects
git fetch origin
```

### Reference Repository

Share objects between repositories.

```bash
# Clone with reference
git clone --reference /path/to/original /path/to/clone

# Add reference to existing repo
git repack -a -d
git clone --reference /path/to/reference https://github.com/user/repo.git

# Dissociate from reference
git repack -a -d
rm -rf .git/objects/info/alternates
```

## Repository Configuration

### Config Scopes

```bash
# System-wide (/etc/gitconfig)
git config --system core.editor vim

# User-level (~/.gitconfig)
git config --global user.name "John Doe"

# Repository-level (.git/config)
git config user.email john@project.com

# Worktree-level (.git/config.worktree)
git config --worktree core.sparseCheckout true

# View all configs with origin
git config --list --show-origin
git config --list --show-scope
```

### Conditional Configuration

**~/.gitconfig:**
```ini
[user]
    name = John Doe
    email = personal@example.com

[includeIf "gitdir:~/work/"]
    path = ~/.gitconfig-work

[includeIf "gitdir:~/projects/opensource/"]
    path = ~/.gitconfig-opensource
```

**~/.gitconfig-work:**
```ini
[user]
    email = john.doe@company.com
```

### Advanced Settings

```bash
# Reuse recorded resolutions
git config rerere.enabled true
git config rerere.autoUpdate true

# Default merge strategy
git config merge.strategy recursive
git config merge.conflictStyle diff3

# Commit signing
git config commit.gpgSign true
git config user.signingKey ABC123

# Push configuration
git config push.default simple
git config push.followTags true
git config push.autoSetupRemote true

# Diff and merge tools
git config diff.tool vimdiff
git config merge.tool meld
git config mergetool.keepBackup false

# Performance settings
git config core.preloadIndex true
git config core.fscache true
git config gc.auto 256

# Security
git config transfer.fsckObjects true
git config receive.fsckObjects true
git config fetch.fsckObjects true
```

### Repository Attributes

**.gitattributes:**
```
# Line endings
* text=auto
*.sh text eol=lf
*.bat text eol=crlf

# Binary files
*.png binary
*.jpg binary
*.pdf binary

# Diff drivers
*.ipynb diff=jupyternotebook
*.json diff=json

# Merge strategies
Makefile merge=union
CHANGELOG.md merge=union

# Export settings
.gitattributes export-ignore
.gitignore export-ignore
tests/ export-ignore

# Language statistics
docs/* linguist-documentation
vendor/* linguist-vendored
*.js linguist-language=JavaScript
```

## Repository Hooks

### Client-Side Hooks

```bash
# Pre-commit: Run before commit
.git/hooks/pre-commit

# Prepare-commit-msg: Modify commit message
.git/hooks/prepare-commit-msg

# Commit-msg: Validate commit message
.git/hooks/commit-msg

# Post-commit: Run after commit
.git/hooks/post-commit

# Pre-push: Run before push
.git/hooks/pre-push

# Post-checkout: Run after checkout
.git/hooks/post-checkout

# Post-merge: Run after merge
.git/hooks/post-merge

# Pre-rebase: Run before rebase
.git/hooks/pre-rebase
```

### Server-Side Hooks

```bash
# Pre-receive: Run before accepting push
.git/hooks/pre-receive

# Update: Run for each branch being updated
.git/hooks/update

# Post-receive: Run after accepting push
.git/hooks/post-receive

# Post-update: Run after all refs updated
.git/hooks/post-update
```

### Hook Examples

**Pre-commit: Run tests**
```bash
#!/bin/bash
npm test
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

**Commit-msg: Enforce format**
```bash
#!/bin/bash
commit_msg_file=$1
commit_msg=$(cat "$commit_msg_file")
pattern="^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+"

if ! echo "$commit_msg" | grep -qE "$pattern"; then
    echo "Invalid commit message format"
    echo "Format: <type>(<scope>): <subject>"
    exit 1
fi
```

**Pre-push: Prevent force push to main**
```bash
#!/bin/bash
while read local_ref local_sha remote_ref remote_sha; do
    if [[ "$remote_ref" == "refs/heads/main" ]] && [[ "$local_sha" != "$remote_sha" ]]; then
        echo "Error: Cannot force push to main branch"
        exit 1
    fi
done
```

## Best Practices

### Repository Organization

1. **Consistent structure**: Use standard directory layout
2. **Clear documentation**: Maintain README, CONTRIBUTING, and docs
3. **Ignore correctly**: Use .gitignore for build artifacts
4. **Small commits**: Atomic, focused changes
5. **Meaningful messages**: Descriptive commit messages
6. **Branch protection**: Protect main branches
7. **Code reviews**: Require reviews before merge
8. **CI/CD integration**: Automated testing and deployment

### Performance Optimization

```bash
# Enable filesystem cache
git config core.fscache true

# Enable preload index
git config core.preloadIndex true

# Use protocol v2
git config protocol.version 2

# Optimize garbage collection
git config gc.auto 256
git config gc.autoPackLimit 50

# Use commit graph
git config core.commitGraph true
git config gc.writeCommitGraph true
git commit-graph write --reachable

# Enable parallel processing
git config pack.threads 0  # Auto-detect CPU count
```

### Security Best Practices

```bash
# Enable fsck on fetch
git config fetch.fsckObjects true
git config receive.fsckObjects true
git config transfer.fsckObjects true

# Sign commits
git config commit.gpgSign true

# Protect against tag replacement
git config receive.denyNonFastForwards true
git config receive.denyDeletes true

# Verify commit signatures
git log --show-signature
git verify-commit HEAD

# Scan for secrets before commit
# Use tools like git-secrets or gitleaks
git secrets --scan
```

### Collaboration Best Practices

1. **Clear branch naming**: Use consistent naming conventions
2. **Protected branches**: Require reviews and tests
3. **Rebase vs merge**: Choose strategy consistently
4. **Force push carefully**: Use --force-with-lease
5. **Clean history**: Squash/rebase before merging
6. **Tag releases**: Use semantic versioning
7. **Document changes**: Maintain CHANGELOG
8. **Code owners**: Use CODEOWNERS file

### Repository Hygiene

```bash
# Regular maintenance
git fetch --prune
git remote prune origin
git gc --auto

# Clean up branches
git branch --merged | grep -v "\*" | xargs -n 1 git branch -d

# Update submodules
git submodule update --remote --merge

# Verify repository health
git fsck --full

# Optimize periodically
git repack -a -d -f --depth=250 --window=250
git gc --aggressive
```

## Troubleshooting

### Common Issues

**Repository corruption:**
```bash
# Verify integrity
git fsck --full

# Recover from corruption
git reflog expire --expire=now --all
git gc --prune=now

# Clone fresh copy if needed
git clone --mirror corrupt-repo fresh-repo
```

**Large repository size:**
```bash
# Find large files
git rev-list --objects --all |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  awk '/^blob/ {print substr($0,6)}' |
  sort -nk2 |
  tail -20

# Remove large files
git filter-repo --strip-blobs-bigger-than 10M

# Reduce pack size
git repack -a -d -f --depth=250 --window=250
```

**Submodule issues:**
```bash
# Reset submodules
git submodule deinit --all -f
git submodule update --init --recursive

# Fix detached HEAD in submodule
cd submodule
git checkout main
cd ..
git add submodule
```

**Merge conflicts in binary files:**
```bash
# Use ours or theirs
git checkout --ours file.bin
git checkout --theirs file.bin

# Configure merge driver
git config merge.ours.driver true
# In .gitattributes:
# *.bin merge=ours
```

**Detached HEAD state:**
```bash
# Create branch from detached HEAD
git checkout -b recovery-branch

# Return to previous branch
git checkout -

# Recover lost commits
git reflog
git checkout -b recovery <commit-hash>
```

### Recovery Operations

**Recover deleted branch:**
```bash
git reflog
git checkout -b recovered-branch <commit-hash>
```

**Recover from hard reset:**
```bash
git reflog
git reset --hard <commit-before-reset>
```

**Undo published commits:**
```bash
# Create revert commit
git revert <commit-hash>

# Or revert multiple commits
git revert <oldest-hash>..<newest-hash>
```

**Fix wrong commit message:**
```bash
# Last commit (not pushed)
git commit --amend -m "Corrected message"

# Older commit
git rebase -i <commit-before>^
# Change "pick" to "reword" for commit
```

## Quick Reference

### Repository Commands

| Command | Description |
|---------|-------------|
| `git init` | Initialize repository |
| `git clone` | Clone repository |
| `git remote` | Manage remotes |
| `git fetch` | Fetch from remote |
| `git pull` | Fetch and merge |
| `git push` | Push to remote |
| `git submodule` | Manage submodules |
| `git subtree` | Manage subtrees |
| `git worktree` | Manage worktrees |
| `git gc` | Garbage collection |
| `git fsck` | Verify repository |
| `git bundle` | Create repository bundle |
| `git filter-repo` | Rewrite history |

### Configuration Levels

| Scope | Flag | File | Priority |
|-------|------|------|----------|
| System | `--system` | `/etc/gitconfig` | Lowest |
| Global | `--global` | `~/.gitconfig` | Medium |
| Local | `--local` | `.git/config` | High |
| Worktree | `--worktree` | `.git/config.worktree` | Highest |

### Repository Health Checklist

- [ ] Repository size is reasonable
- [ ] No large binary files in history
- [ ] .gitignore is comprehensive
- [ ] Branches are up to date
- [ ] Remote references are clean
- [ ] Hooks are configured correctly
- [ ] Tests pass on all branches
- [ ] Documentation is current
- [ ] Security scanning enabled
- [ ] Backup strategy in place

Effective repository management ensures smooth collaboration, optimal performance, and long-term maintainability of your projects.
