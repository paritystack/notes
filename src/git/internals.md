# Git Internals

## Overview

Git is often described as a "content-addressable filesystem with a VCS user interface on top." Understanding Git's internal architecture reveals how it efficiently stores data, tracks changes, and enables powerful version control operations. This guide explores the plumbing commands, internal data structures, and core concepts that make Git work.

**Why Learn Git Internals?**
- Debug complex issues more effectively
- Understand what commands actually do
- Recover from disasters
- Optimize repository performance
- Build custom Git tools and automation

## The .git Directory

Every Git repository has a `.git` directory containing all Git metadata and objects.

### Directory Structure

```bash
.git/
├── HEAD              # Points to current branch
├── config            # Repository-specific configuration
├── description       # Repository description (for GitWeb)
├── index             # Staging area (binary file)
├── hooks/            # Client and server-side hook scripts
├── info/             # Global exclude file and refs
│   └── exclude       # gitignore patterns not in .gitignore
├── objects/          # All content: commits, trees, blobs, tags
│   ├── pack/         # Packfiles for efficient storage
│   └── info/         # Object info and packs
├── refs/             # References (branches and tags)
│   ├── heads/        # Local branches
│   ├── remotes/      # Remote-tracking branches
│   └── tags/         # Tags
├── logs/             # Reflog information
│   ├── HEAD          # HEAD history
│   └── refs/         # Branch history
└── packed-refs       # Packed references for performance
```

### Exploring .git Directory

```bash
# Navigate to .git
cd .git

# View HEAD (current branch pointer)
cat HEAD
# Output: ref: refs/heads/main

# View current branch
cat refs/heads/main
# Output: a3f2b1c... (commit SHA-1)

# View remote branch
cat refs/remotes/origin/main

# List all objects
find objects/ -type f
```

## Git Objects: The Building Blocks

Git stores everything as objects identified by SHA-1 hashes. There are four object types:

1. **Blob** - File content
2. **Tree** - Directory structure
3. **Commit** - Snapshot with metadata
4. **Tag** - Annotated tag with metadata

### Object Storage

Objects are stored in `.git/objects/`:
- First 2 characters of SHA-1 = subdirectory
- Remaining 38 characters = filename
- Content is zlib-compressed

```bash
# Example: Object a3f2b1c4...
# Stored at: .git/objects/a3/f2b1c4...
```

### 1. Blob Objects

Blobs store file content (data only, no filename or metadata).

```bash
# Create a blob manually (plumbing)
echo "Hello, Git!" | git hash-object -w --stdin
# Output: 8ab686eafeb1f44702738c8b0f24f2567c36da6d

# -w = write to object database
# --stdin = read from standard input

# View blob content
git cat-file -p 8ab686eafeb1f44702738c8b0f24f2567c36da6d
# Output: Hello, Git!

# Check object type
git cat-file -t 8ab686ea
# Output: blob

# View object size
git cat-file -s 8ab686ea
# Output: 12
```

**Creating blobs from files:**

```bash
# Create a file
echo "Git internals are fascinating" > test.txt

# Hash and store the file
git hash-object -w test.txt
# Output: 2c8b4e3b7c1a9f...

# The blob is now in .git/objects/
# File content is stored, but filename is NOT
```

### 2. Tree Objects

Trees represent directory structure, mapping filenames to blobs and other trees.

```bash
# View a tree object
git cat-file -p main^{tree}
# Output:
# 100644 blob a3f2b1c...    README.md
# 100644 blob 7e4d3a2...    index.js
# 040000 tree 9c1f5b8...    src

# Tree entries format:
# <mode> <type> <sha-1> <filename>
```

**File Modes:**
- `100644` - Normal file
- `100755` - Executable file
- `120000` - Symbolic link
- `040000` - Directory (tree)
- `160000` - Gitlink (submodule)

**Creating a tree manually:**

```bash
# Create a tree from index
git write-tree
# Output: 9c1f5b8a... (tree SHA-1)

# Add files to index first
git update-index --add --cacheinfo 100644 \
  a3f2b1c4... README.md

git update-index --add --cacheinfo 100644 \
  7e4d3a2b... index.js

# Write tree from current index
git write-tree
```

**Reading tree contents:**

```bash
# List tree contents recursively
git ls-tree -r -t main^{tree}
# -r = recursive
# -t = show trees as well

# Pretty print tree structure
git ls-tree --abbrev main^{tree}
```

### 3. Commit Objects

Commits point to a tree (snapshot) and contain metadata.

```bash
# View commit object
git cat-file -p HEAD
# Output:
# tree 9c1f5b8a...
# parent a3f2b1c4...
# author John Doe <john@example.com> 1234567890 -0500
# committer John Doe <john@example.com> 1234567890 -0500
#
# Commit message here
```

**Commit Structure:**
- `tree` - Points to root tree (project snapshot)
- `parent` - Previous commit(s); merge commits have multiple
- `author` - Who wrote the code (name, email, timestamp)
- `committer` - Who committed (may differ from author)
- Commit message

**Creating a commit manually:**

```bash
# Create a commit (plumbing)
echo "Initial commit" | git commit-tree 9c1f5b8a
# Output: b4e3c2d1... (commit SHA-1)

# Create commit with parent
echo "Second commit" | git commit-tree 7a2b3c4d -p b4e3c2d1

# Update branch to point to new commit
git update-ref refs/heads/main b4e3c2d1
```

### 4. Tag Objects

Annotated tags are objects containing metadata about a tag.

```bash
# Create annotated tag
git tag -a v1.0 -m "Version 1.0"

# View tag object
git cat-file -p v1.0
# Output:
# object a3f2b1c4...
# type commit
# tag v1.0
# tagger John Doe <john@example.com> 1234567890 -0500
#
# Version 1.0
```

**Lightweight vs Annotated Tags:**

```bash
# Lightweight tag (just a ref)
git tag v1.0-light
cat .git/refs/tags/v1.0-light
# Output: a3f2b1c4... (points directly to commit)

# Annotated tag (object)
git tag -a v1.0 -m "Release"
cat .git/refs/tags/v1.0
# Output: b7e8f3a... (points to tag object)
```

## Content-Addressable Storage

Git uses SHA-1 hashing to create content-addressable storage.

### How SHA-1 Works in Git

```bash
# Git computes SHA-1 of:
# "blob <size>\0<content>"

# Example calculation
content="Hello, Git!"
size=${#content}
(printf "blob %s\0" $size; echo -n "$content") | sha1sum
# Output: 8ab686eafeb1f44702738c8b0f24f2567c36da6d

# This matches what git hash-object produces
echo "Hello, Git!" | git hash-object --stdin
```

### Properties of Content-Addressable Storage

1. **Deduplication**: Identical content = same hash = stored once
2. **Integrity**: SHA-1 acts as checksum; corruption is detectable
3. **Immutability**: Can't change content without changing hash
4. **Efficient**: Easy to check if object exists (hash lookup)

```bash
# Example: Two files with identical content
echo "Same content" > file1.txt
echo "Same content" > file2.txt

# Both produce same blob
git hash-object file1.txt  # abc123...
git hash-object file2.txt  # abc123... (identical!)

# Git stores content only once
```

## File Tracking and the Index

The **index** (staging area) is a binary file at `.git/index` that serves as a staging area between the working directory and repository.

### The Three States of Files

```
┌─────────────────┐    git add    ┌─────────────────┐   git commit   ┌─────────────────┐
│  Working Dir    │──────────────→│  Staging Area   │───────────────→│   Repository    │
│   (modified)    │               │    (staged)     │                │   (committed)   │
└─────────────────┘               └─────────────────┘                └─────────────────┘
        ↑                                                                     │
        └─────────────────────────────── git checkout ────────────────────────┘
```

### File States

1. **Untracked**: Not in index or last commit
2. **Unmodified**: In repository, unchanged
3. **Modified**: Changed since last commit
4. **Staged**: Marked for next commit

```bash
# File lifecycle diagram
Untracked ──add──→ Staged ──commit──→ Unmodified
                      ↑                    │
                      │                    │ edit
                      │                    ↓
                      └────────────────Modified
```

### Viewing the Index

```bash
# View index contents
git ls-files --stage
# Output:
# 100644 a3f2b1c... 0    README.md
# 100644 7e4d3a2... 0    src/index.js
# 100644 9f8e7d6... 0    package.json

# Format: <mode> <sha-1> <stage> <filename>
# stage: 0 = normal, 1-3 = conflict resolution stages
```

**Index Stages (for merge conflicts):**
- Stage 0: Normal entry
- Stage 1: Common ancestor version
- Stage 2: "ours" (current branch)
- Stage 3: "theirs" (merging branch)

```bash
# During merge conflict
git ls-files --stage
# 100644 a1b2c3... 1    conflicted.txt  (base)
# 100644 d4e5f6... 2    conflicted.txt  (ours)
# 100644 g7h8i9... 3    conflicted.txt  (theirs)
```

### Working with the Index

```bash
# Add file to index
git update-index --add --cacheinfo 100644 a3f2b1c README.md

# Remove from index (keep in working dir)
git update-index --force-remove README.md

# Refresh index (update stat info)
git update-index --refresh

# Show index and working tree differences
git diff-files
# Shows files modified in working dir

# Show index and repository differences
git diff-index --cached HEAD
# Shows staged changes
```

### How git add Works Internally

```bash
# When you run: git add file.txt

# 1. Git computes SHA-1 of file content
hash=$(git hash-object -w file.txt)

# 2. Stores blob in .git/objects/
# (Already done by -w flag above)

# 3. Updates index with new hash
git update-index --add --cacheinfo 100644 $hash file.txt

# This is what git add does behind the scenes!
```

## Refs: Pointers to Commits

**References (refs)** are human-readable names that point to commits. They're stored in `.git/refs/`.

### Types of Refs

1. **Heads** (branches): `.git/refs/heads/`
2. **Tags**: `.git/refs/tags/`
3. **Remotes**: `.git/refs/remotes/`

```bash
# View a ref (just a file with commit SHA-1)
cat .git/refs/heads/main
# Output: a3f2b1c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0

# All refs are just text files!
```

### HEAD: The Current Reference

**HEAD** is a symbolic reference pointing to the current branch.

```bash
# View HEAD
cat .git/HEAD
# Output: ref: refs/heads/main

# HEAD points to a branch, which points to a commit
cat .git/refs/heads/main
# Output: a3f2b1c...
```

**Normal HEAD (attached):**
```
HEAD → refs/heads/main → commit a3f2b1c
```

**Detached HEAD:**
```
HEAD → commit a3f2b1c (no branch)
```

### Detached HEAD State

```bash
# Checkout specific commit
git checkout a3f2b1c
# Warning: You are in 'detached HEAD' state

# HEAD now points directly to commit
cat .git/HEAD
# Output: a3f2b1c... (no longer "ref: refs/heads/...")

# Any commits made here are "orphaned" unless you create a branch
git switch -c new-branch  # Attach HEAD to new branch
```

### Symbolic References

```bash
# HEAD is a symbolic ref
git symbolic-ref HEAD
# Output: refs/heads/main

# Change HEAD to point to different branch
git symbolic-ref HEAD refs/heads/develop
# Now on develop branch (without checking out files)

# Read the ref HEAD points to
git symbolic-ref HEAD
git rev-parse HEAD  # Get the commit SHA-1
```

### Special Refs

- **HEAD**: Current commit/branch
- **ORIG_HEAD**: Previous HEAD (before risky operations)
- **FETCH_HEAD**: Last fetched branch
- **MERGE_HEAD**: Commit being merged
- **CHERRY_PICK_HEAD**: Commit being cherry-picked

```bash
# ORIG_HEAD is set by commands that move HEAD
git reset --hard HEAD~1  # ORIG_HEAD now points to previous HEAD

# Undo the reset
git reset --hard ORIG_HEAD

# MERGE_HEAD during merge
git merge feature-branch
# .git/MERGE_HEAD exists during merge conflict
cat .git/MERGE_HEAD  # Shows commit being merged
```

### Creating and Managing Refs

```bash
# Create a branch (low-level)
git update-ref refs/heads/new-branch a3f2b1c

# This is what git branch does!
# Equivalent to:
echo "a3f2b1c..." > .git/refs/heads/new-branch

# Delete a ref
git update-ref -d refs/heads/old-branch

# List all refs
git for-each-ref
# Output:
# a3f2b1c... commit refs/heads/main
# b4e3c2d... commit refs/heads/feature
# 7a2b3c4... commit refs/remotes/origin/main
# 9f8e7d6... tag    refs/tags/v1.0

# Format output
git for-each-ref --format='%(refname:short) %(objecttype) %(objectname:short)'
```

### Packed References

For performance, Git can pack refs into `.git/packed-refs`.

```bash
# View packed refs
cat .git/packed-refs
# Output:
# # pack-refs with: peeled fully-peeled sorted
# a3f2b1c... refs/heads/main
# b4e3c2d... refs/remotes/origin/main
# 7a2b3c4... refs/tags/v1.0
# ^9f8e7d6... (peeled tag - points to commit, not tag object)

# Pack refs manually
git pack-refs --all --prune

# Loose refs take precedence over packed refs
```

## Plumbing vs Porcelain Commands

Git commands are divided into two categories:

- **Porcelain**: High-level user-friendly commands (`git commit`, `git push`)
- **Plumbing**: Low-level commands that manipulate Git internals

### Why Plumbing Commands?

1. **Automation**: Build scripts and tools
2. **Understanding**: Learn how Git works
3. **Recovery**: Fix broken repositories
4. **Debugging**: Investigate issues

### Essential Plumbing Commands

#### Object Inspection

```bash
# cat-file: View object content
git cat-file -t a3f2b1c     # Type (blob, tree, commit, tag)
git cat-file -s a3f2b1c     # Size
git cat-file -p a3f2b1c     # Pretty-print content
git cat-file blob a3f2b1c   # View blob content

# rev-parse: Parse revisions
git rev-parse HEAD          # Full SHA-1 of HEAD
git rev-parse --short HEAD  # Short SHA-1
git rev-parse main          # Resolve branch to commit
git rev-parse HEAD~3        # Three commits before HEAD

# ls-tree: List tree contents
git ls-tree HEAD            # Root tree
git ls-tree -r HEAD         # Recursive
git ls-tree HEAD src/       # Specific directory
```

#### Object Creation

```bash
# hash-object: Create blob
echo "content" | git hash-object -w --stdin
git hash-object -w file.txt

# mktree: Create tree from stdin
# Format: <mode> SP <type> SP <sha1> TAB <filename>
cat | git mktree << EOF
100644 blob a3f2b1c... file1.txt
100644 blob b4e3c2d... file2.txt
040000 tree 7a2b3c4... subdir
EOF

# commit-tree: Create commit
echo "Commit message" | git commit-tree 9c1f5b8 -p a3f2b1c

# write-tree: Create tree from index
git write-tree
```

#### Reference Management

```bash
# update-ref: Create/update references
git update-ref refs/heads/test a3f2b1c
git update-ref -d refs/heads/test  # Delete

# symbolic-ref: Manage symbolic refs
git symbolic-ref HEAD refs/heads/main

# for-each-ref: Iterate over refs
git for-each-ref refs/heads/
git for-each-ref --format='%(refname)' refs/tags/
```

#### Index Manipulation

```bash
# update-index: Modify index
git update-index --add --cacheinfo 100644 a3f2b1c file.txt
git update-index --remove file.txt
git update-index --refresh

# ls-files: Show index contents
git ls-files                  # All tracked files
git ls-files --stage          # With hash and mode
git ls-files --deleted        # Deleted in working dir
git ls-files --modified       # Modified in working dir
git ls-files --others         # Untracked files

# read-tree: Read tree into index
git read-tree HEAD            # Reset index to HEAD
git read-tree --prefix=sub/ HEAD  # Read into subdirectory
```

#### Comparison and Diffing

```bash
# diff-tree: Compare trees
git diff-tree HEAD HEAD~1     # Compare commits
git diff-tree -r HEAD HEAD~1  # Recursive

# diff-index: Compare index
git diff-index HEAD           # Index vs HEAD
git diff-index --cached HEAD  # Staged changes

# diff-files: Compare working dir
git diff-files                # Working dir vs index
```

### Building Porcelain with Plumbing

**Example: Implementing `git add` with plumbing**

```bash
#!/bin/bash
# add.sh - Simplified git add implementation

file=$1

# 1. Hash and store file content
hash=$(git hash-object -w "$file")

# 2. Update index
git update-index --add --cacheinfo 100644 "$hash" "$file"

echo "Added $file (hash: $hash)"
```

**Example: Implementing `git commit` with plumbing**

```bash
#!/bin/bash
# commit.sh - Simplified git commit implementation

message=$1

# 1. Create tree from current index
tree=$(git write-tree)

# 2. Get parent commit
parent=$(git rev-parse HEAD)

# 3. Create commit object
commit=$(echo "$message" | git commit-tree "$tree" -p "$parent")

# 4. Update branch ref
git update-ref refs/heads/$(git rev-parse --abbrev-ref HEAD) "$commit"

echo "Created commit $commit"
```

## Commit Ancestry and References

Git uses special syntax to refer to commits relative to each other.

### Ancestry References

```bash
# Parent references
HEAD~1      # First parent (same as HEAD^)
HEAD~2      # Second parent (grandparent)
HEAD~3      # Third parent (great-grandparent)

# Multiple parents (merge commits)
HEAD^1      # First parent
HEAD^2      # Second parent (merged branch)

# Combining
HEAD~2^2    # Second parent of grandparent
```

**Difference between ~ and ^:**
- `~` always follows first parent
- `^` can select which parent

```bash
# Merge commit example
    A
   / \
  B   C
  |
  D

# A~1 = B (first parent)
# A^1 = B (first parent)
# A^2 = C (second parent)
# A~2 = D (grandparent via first parent)
```

### Commit Ranges

```bash
# Double dot: Commits in B not in A
git log A..B
# Example: main..feature (commits in feature not in main)

# Triple dot: Commits in A or B, but not both
git log A...B
# Example: main...feature (symmetric difference)

# All ancestors of B excluding A
git log ^A B
git log A..B  # Equivalent

# Multiple exclusions
git log ^A ^B C
# Commits in C but not in A or B
```

**Practical examples:**

```bash
# View commits in feature branch not in main
git log main..feature

# View commits that will be pushed
git log origin/main..HEAD

# View commits in current branch since branching from main
git log main..HEAD

# Show what changed between two branches
git log --oneline main...feature

# Find merge base
git merge-base main feature
```

### Refspecs

**Refspecs** define mappings between remote and local refs.

```bash
# View refspec
git config --get-regexp remote.origin
# Output:
# remote.origin.url https://github.com/user/repo.git
# remote.origin.fetch +refs/heads/*:refs/remotes/origin/*

# Refspec format:
# [+]<source>:<destination>
# + = force update
```

**Fetch refspec:** `+refs/heads/*:refs/remotes/origin/*`
- Maps all remote branches to local remote-tracking branches
- `refs/heads/main` → `refs/remotes/origin/main`

**Push refspec:** `refs/heads/*:refs/heads/*`
- Maps local branches to remote branches
- `refs/heads/main` → `refs/heads/main` (on remote)

```bash
# Custom refspec examples

# Fetch only main branch
git config remote.origin.fetch refs/heads/main:refs/remotes/origin/main

# Fetch all branches (default)
git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'

# Push branch to different name
git push origin local-branch:remote-branch

# Push to different ref
git push origin HEAD:refs/heads/new-branch

# Delete remote branch
git push origin :branch-to-delete
# Or
git push origin --delete branch-to-delete

# Fetch pull request (GitHub)
git fetch origin pull/123/head:pr-123
```

## The Reflog

The **reflog** records when refs (HEAD, branches) were updated. It's essential for recovery.

### Understanding Reflog

```bash
# View HEAD reflog
git reflog
# Output:
# a3f2b1c (HEAD -> main) HEAD@{0}: commit: Add feature
# b4e3c2d HEAD@{1}: commit: Fix bug
# 7a2b3c4 HEAD@{2}: checkout: moving from dev to main

# Reflog for specific ref
git reflog show main
git reflog show origin/main
```

### Reflog Syntax

```bash
# @{n} - nth prior value
HEAD@{0}    # Current HEAD
HEAD@{1}    # Previous HEAD
HEAD@{2}    # Two steps back

# @{time} - Value at specific time
HEAD@{5.minutes.ago}
HEAD@{yesterday}
HEAD@{2.days.ago}
HEAD@{2023-01-01}

# Examples
git show HEAD@{5}             # Show 5th prior HEAD
git diff HEAD@{0} HEAD@{1}    # Compare current vs previous
git log -g HEAD               # Show reflog as log
```

### Recovery with Reflog

```bash
# Scenario: Accidentally reset hard
git reset --hard HEAD~3  # Oops!

# Find commit before reset
git reflog
# a3f2b1c HEAD@{0}: reset: moving to HEAD~3
# b4e3c2d HEAD@{1}: commit: Lost commit

# Recover
git reset --hard HEAD@{1}
# Or
git reset --hard b4e3c2d

# Recover deleted branch
git reflog --all  # Show all refs
git branch recovered-branch a3f2b1c
```

**Scenario: Recover from bad rebase**

```bash
# Before rebase
git log --oneline
# a3f2b1c (HEAD -> feature) Feature work
# b4e3c2d More feature work
# 7a2b3c4 (main) Main work

# Bad interactive rebase (dropped commits)
git rebase -i main
# Accidentally deleted commits!

# Find commits in reflog
git reflog
# 9f8e7d6 HEAD@{0}: rebase -i: finish
# a3f2b1c HEAD@{1}: rebase -i: start

# Reset to before rebase
git reset --hard HEAD@{1}
```

### Reflog Expiration

```bash
# Reflogs are temporary (default: 90 days)
# Unreachable commits expire after 30 days

# View reflog expiration config
git config --get gc.reflogExpire        # Default: 90 days
git config --get gc.reflogExpireUnreachable  # Default: 30 days

# Manually expire reflog
git reflog expire --expire=now --all
git gc --prune=now

# Keep reflog forever (not recommended)
git config gc.reflogExpire never
```

## Branches Under the Hood

Branches are simply refs pointing to commits. Understanding this reveals Git's power.

### What is a Branch?

```bash
# A branch is just a file containing a commit hash
cat .git/refs/heads/main
# Output: a3f2b1c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0

# That's it! Just 40 bytes (or less in packed-refs)
```

### Creating Branches with Plumbing

```bash
# Porcelain
git branch new-feature

# Plumbing equivalent
git update-ref refs/heads/new-feature HEAD

# Or even more manual
echo $(git rev-parse HEAD) > .git/refs/heads/new-feature
```

### Switching Branches

```bash
# Porcelain
git checkout main

# Plumbing steps:
# 1. Update HEAD
git symbolic-ref HEAD refs/heads/main

# 2. Update index and working directory
git read-tree --reset -u HEAD

# 3. That's it!
```

### Merging: Fast-Forward vs Three-Way

**Fast-forward merge:**
```
Before:
  main     feature
    ↓         ↓
A - B - C - D

After (git merge feature):
  main/feature
       ↓
A - B - C - D
```

```bash
# Fast-forward is just updating the ref
git update-ref refs/heads/main $(git rev-parse feature)
```

**Three-way merge:**
```
Before:
      C (main)
     /
A - B
     \
      D (feature)

After (git merge feature):
      C - M (main)
     /   /
A - B - D
```

```bash
# Three-way merge creates new commit with two parents
# Parents: current HEAD and merged branch

# Plumbing equivalent:
tree=$(git write-tree)  # From merge resolution
commit=$(echo "Merge message" | git commit-tree $tree \
  -p $(git rev-parse HEAD) \
  -p $(git rev-parse feature))
git update-ref refs/heads/main $commit
```

## Remote Tracking

Understanding how Git tracks remote branches.

### Remote-Tracking Branches

```bash
# Remote-tracking branches are refs under refs/remotes/
ls -la .git/refs/remotes/origin/
# main
# develop
# feature-123

# They're just refs, like local branches
cat .git/refs/remotes/origin/main
# b4e3c2d1... (commit hash)
```

### How git fetch Works

```bash
# Porcelain
git fetch origin

# Plumbing steps:
# 1. Connect to remote
# 2. Receive pack of new objects
# 3. Store objects in .git/objects/
# 4. Update refs/remotes/origin/* refs

# Fetch specific branch
git fetch origin main:refs/remotes/origin/main
```

### How git pull Works

```bash
# git pull = git fetch + git merge

# Equivalent to:
git fetch origin
git merge origin/main

# Or with rebase:
git fetch origin
git rebase origin/main
```

### How git push Works

```bash
# Porcelain
git push origin main

# Plumbing steps:
# 1. Check if fast-forward possible
# 2. Pack objects not on remote
# 3. Send pack to remote
# 4. Remote updates refs/heads/main

# Push creates commits on remote, then updates ref
# Equivalent refspec:
git push origin refs/heads/main:refs/heads/main
```

### Tracking Branches

```bash
# Set upstream branch
git branch --set-upstream-to=origin/main main

# This adds to .git/config:
# [branch "main"]
#     remote = origin
#     merge = refs/heads/main

# View tracking relationship
git branch -vv
# main  a3f2b1c [origin/main] Latest commit

# Remote tracking allows:
git pull   # Knows to pull from origin/main
git push   # Knows to push to origin/main
```

## Pack Files and Storage Optimization

Git uses pack files to compress objects efficiently.

### Loose vs Packed Objects

**Loose objects:**
- Individual files in `.git/objects/ab/cdef...`
- Zlib-compressed
- One object per file
- Fast to create, slower to access in bulk

**Packed objects:**
- Combined into `.git/objects/pack/pack-*.pack`
- Delta-compressed (stores differences)
- Accompanied by `.idx` index file
- Slower to create, much faster to access

### Viewing Object Storage

```bash
# Count objects
git count-objects -v
# Output:
# count: 150           # Loose objects
# size: 600            # KB
# in-pack: 3500        # Packed objects
# packs: 1             # Number of pack files
# size-pack: 1200      # KB in packs
# prune-packable: 0
# garbage: 0
# size-garbage: 0

# List pack files
ls -lh .git/objects/pack/
# pack-abc123.idx
# pack-abc123.pack
```

### Pack File Structure

```bash
# .pack file: Contains compressed objects
# .idx file: Index for finding objects in pack

# Verify pack
git verify-pack -v .git/objects/pack/pack-*.idx
# Output:
# a3f2b1c blob   150 140 12
# b4e3c2d blob   200 185 152
# 7a2b3c4 commit 250 235 337
# ...
# non delta: 150 objects
# chain length = 10: 50 objects
```

### Delta Compression

Git stores deltas (differences) to save space:

```bash
# Example: Two similar files
# version1.txt: "Hello World"
# version2.txt: "Hello World!\nNew line"

# Git stores:
# - Full version2.txt (base)
# - Delta: version1 relative to version2

# Verify pack shows delta chains
git verify-pack -v .git/objects/pack/pack-*.idx | grep chain
```

### Garbage Collection

```bash
# Manual garbage collection
git gc
# - Packs loose objects
# - Removes unreachable objects
# - Optimizes repository

# Aggressive GC (slow but thorough)
git gc --aggressive
# More thorough delta compression

# Prune unreachable objects
git prune
# Remove objects not reachable from any ref

# Prune everything older than 2 weeks
git prune --expire=2.weeks.ago

# Automatic GC
git config gc.auto 6700  # Auto-gc after 6700 loose objects
git config gc.autopacklimit 50  # Auto-gc after 50 pack files
```

### Optimizing Repository

```bash
# Create pack file from scratch
git repack -a -d -f
# -a = all objects
# -d = remove redundant packs
# -f = force

# Aggressive repacking
git repack -a -d -f --depth=250 --window=250

# Reduce repository size
git gc --aggressive --prune=now

# Clone with shallow history (for large repos)
git clone --depth 1 <url>
# Only most recent commit
```

## Advanced Internals Topics

### The Index File Format

The index is a binary file with this structure:

```
Header (12 bytes):
- Signature: "DIRC" (DIrectory Cache)
- Version: 2, 3, or 4
- Number of entries

Entry (variable length):
- ctime/mtime metadata
- Device/inode
- Mode (file permissions)
- UID/GID
- File size
- SHA-1 (20 bytes)
- Flags (name length, stage)
- File name

Extensions:
- Tree cache
- Resolve undo
- etc.
```

```bash
# Dump index in human-readable format
git ls-files --stage --debug
```

### Object Database Deep Dive

```bash
# Find all objects
find .git/objects -type f

# Object file structure:
# - zlib compressed
# - Header: "<type> <size>\0"
# - Content

# Decompress object manually (example)
printf "\x1f\x8b\x08\x00\x00\x00\x00\x00" | \
  cat - .git/objects/ab/cdef... | \
  gunzip

# Or use Git's plumbing
git cat-file -p abcdef
```

### Git Hooks and Plumbing

Hooks are scripts in `.git/hooks/` that run at specific points.

```bash
# Example: pre-commit hook using plumbing
# .git/hooks/pre-commit

#!/bin/bash
# Check for TODO comments in staged files

for file in $(git diff-index --cached --name-only HEAD); do
  if git cat-file -p :0:$file | grep -q "TODO"; then
    echo "Error: TODO found in $file"
    exit 1
  fi
done
```

### Inspecting Repository Health

```bash
# Check repository integrity
git fsck
# - Verifies object connectivity
# - Checks for corruption
# - Reports dangling/unreachable objects

# Full check
git fsck --full
# Output:
# Checking object directories: 100% (256/256), done.
# Checking objects: 100% (3456/3456), done.
# dangling commit abc123...

# Find large objects
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | \
  awk '/^blob/ {print substr($0,6)}' | \
  sort -nk2 | \
  tail -20
```

## Practical Plumbing Use Cases

### 1. Find When File Was Deleted

```bash
# Find when file was deleted
git log --all --full-history -- deleted-file.txt

# Using plumbing
git rev-list --all -- deleted-file.txt | while read commit; do
  if ! git ls-tree -r $commit | grep -q deleted-file.txt; then
    echo "Deleted in: $commit"
    break
  fi
done
```

### 2. Extract File from History

```bash
# Get file from specific commit
commit="abc123"
file="path/to/file.txt"

# Find blob hash
blob=$(git ls-tree $commit $file | awk '{print $3}')

# Extract content
git cat-file blob $blob > recovered-file.txt
```

### 3. Rewrite History to Remove Sensitive Data

```bash
# Remove file from all commits (using plumbing concepts)
git filter-branch --tree-filter 'rm -f passwords.txt' HEAD

# Or with plumbing (manual approach for understanding):
git rev-list --all | while read commit; do
  # Get tree
  tree=$(git rev-parse $commit^{tree})

  # Create new tree without sensitive file
  # (Complex - requires manual tree manipulation)

  # Create new commit
  new_commit=$(git commit-tree ...)

  # Update refs
  git update-ref ...
done
```

### 4. Create Orphan Branch

```bash
# Porcelain
git checkout --orphan new-root

# Plumbing equivalent
# Create empty tree
empty_tree=$(git hash-object -t tree /dev/null)

# Create first commit
commit=$(echo "Initial" | git commit-tree $empty_tree)

# Create branch
git update-ref refs/heads/new-root $commit

# Switch to branch
git symbolic-ref HEAD refs/heads/new-root
git reset --hard
```

### 5. Analyze Repository Statistics

```bash
# Count commits per author
git rev-list --all --pretty=format:'%an' | \
  grep -v '^commit' | \
  sort | uniq -c | sort -nr

# Find largest commits
git rev-list --objects --all | \
  git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' | \
  grep '^commit' | \
  sort -k2 -n | \
  tail -10

# List all files ever committed
git rev-list --objects --all | \
  grep -v '^commit' | \
  cut -d' ' -f2- | \
  sort -u
```

## Debugging with Plumbing

### Trace Git Commands

```bash
# See what Git is doing
GIT_TRACE=1 git commit -m "Test"
# Output shows underlying commands

# Trace pack operations
GIT_TRACE_PACK_ACCESS=1 git fetch

# Trace performance
GIT_TRACE_PERFORMANCE=1 git status
```

### Verbose Object Information

```bash
# Find object by content
echo "search content" | git hash-object --stdin

# Check if object exists
git cat-file -e abc123 && echo "exists"

# Batch check objects
echo -e "abc123\ndef456\n789abc" | \
  git cat-file --batch-check

# Follow rename history
git log --follow --all -- file.txt
```

## Best Practices

1. **Don't Modify .git Manually**
   - Use plumbing commands instead
   - Prevents corruption

2. **Understand Before Using**
   - Plumbing commands can be destructive
   - Test in disposable repositories first

3. **Use Reflog for Safety**
   - Reflog can recover from mistakes
   - Keep reflog enabled

4. **Regular Maintenance**
   - Run `git gc` periodically
   - Check health with `git fsck`

5. **Backup Before Experiments**
   - `cp -r .git .git.backup`
   - Or use separate clone

6. **Learn Incrementally**
   - Start with inspection commands
   - Progress to modification commands
   - Master recovery techniques

## Summary

Git's internals are elegant and understandable:

- **Objects** (blob, tree, commit, tag) are the foundation
- **Refs** are pointers to commits
- **Index** bridges working directory and repository
- **Plumbing** commands manipulate these primitives directly
- **Pack files** optimize storage
- **Reflog** enables recovery

Understanding internals empowers you to:
- Debug complex issues
- Recover from disasters
- Build custom automation
- Optimize repository performance
- Contribute to Git itself

The next time a porcelain command behaves unexpectedly, you'll understand why and how to fix it using plumbing commands.

## Resources

### Official Documentation
- [Git Internals - Git Book](https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain)
- [Git Plumbing Commands](https://git-scm.com/docs/git#_low_level_commands_plumbing)
- [Git Repository Layout](https://git-scm.com/docs/gitrepository-layout)

### Advanced Topics
- [Git Pack Format](https://git-scm.com/docs/pack-format)
- [Git Index Format](https://git-scm.com/docs/index-format)
- [Git Protocol](https://git-scm.com/docs/protocol-v2)

### Tools
- [git-sizer](https://github.com/github/git-sizer) - Analyze repository size
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) - Remove sensitive data
- [git-filter-repo](https://github.com/newren/git-filter-repo) - Rewrite history

### Visualization
- [Git Object Model Visualizer](https://git-school.github.io/visualizing-git/)
- [Explain Git with D3](https://onlywei.github.io/explain-git-with-d3/)

Remember: With great power (plumbing commands) comes great responsibility. Always have backups!
