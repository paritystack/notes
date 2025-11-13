# Git Commands Reference

Comprehensive reference of Git commands organized by category.

## Repository Setup

### Initialize Repository

```bash
# Create new Git repository
git init

# Initialize with specific branch name
git init -b main

# Create bare repository (for remote)
git init --bare
```

### Clone Repository

```bash
# Clone repository
git clone <repository_url>

# Clone to specific directory
git clone <repository_url> <directory>

# Clone specific branch
git clone -b <branch> <repository_url>

# Shallow clone (limited history)
git clone --depth 1 <repository_url>

# Clone with submodules
git clone --recursive <repository_url>
```

## Configuration

### User Settings

```bash
# Set user name
git config --global user.name "Your Name"

# Set user email
git config --global user.email "your.email@example.com"

# View user name
git config user.name

# View user email
git config user.email
```

### Repository Settings

```bash
# Set local config (repository-specific)
git config user.name "Your Name"

# Set editor
git config --global core.editor "vim"

# Set default branch name
git config --global init.defaultBranch main

# Enable color output
git config --global color.ui auto

# Set merge strategy
git config --global pull.rebase false

# View all config
git config --list

# View specific config
git config <key>

# Edit config file
git config --global --edit
```

### Aliases

```bash
# Create alias
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --graph --oneline --all'
```

## Basic Operations

### Status and Information

```bash
# Show working tree status
git status

# Short status
git status -s

# Show status with branch info
git status -sb

# List all tracked files
git ls-files

# Show repository info
git remote show origin
```

### Add Files

```bash
# Add specific file
git add <file>

# Add all files
git add .

# Add all files in directory
git add <directory>/

# Add by pattern
git add *.js

# Add interactively
git add -i

# Add patch (selective staging)
git add -p

# Add all (including deleted)
git add -A

# Add modified and deleted (not new)
git add -u
```

### Commit Changes

```bash
# Commit staged changes
git commit -m "Commit message"

# Commit with detailed message (opens editor)
git commit

# Stage all tracked files and commit
git commit -am "Commit message"

# Amend last commit
git commit --amend

# Amend without changing message
git commit --amend --no-edit

# Amend and change author
git commit --amend --author="Name <email>"

# Empty commit (no changes)
git commit --allow-empty -m "Empty commit"

# Commit with specific date
git commit --date="2024-01-01" -m "Message"
```

### Remove and Move Files

```bash
# Remove file from working directory and staging
git rm <file>

# Remove file from staging only (keep in working directory)
git rm --cached <file>

# Remove directory
git rm -r <directory>

# Move/rename file
git mv <old_name> <new_name>
```

## Viewing History

### Log

```bash
# View commit history
git log

# Compact one-line log
git log --oneline

# Graph view
git log --graph --oneline --all

# Decorate with branch/tag names
git log --decorate

# Pretty format
git log --pretty=format:"%h - %an, %ar : %s"

# Limit number of commits
git log -n 5
git log -5

# Show commits by author
git log --author="John"

# Show commits in date range
git log --since="2 weeks ago"
git log --after="2024-01-01"
git log --until="2024-12-31"
git log --before="2024-12-31"

# Show file statistics
git log --stat

# Show detailed patch
git log -p

# Show commits affecting specific file
git log -- <file>

# Search commit messages
git log --grep="fix bug"

# Show commits that added/removed specific text
git log -S "function_name"

# Show commits by committer (not author)
git log --committer="John"

# Show merge commits only
git log --merges

# Show non-merge commits
git log --no-merges

# Show first parent only
git log --first-parent
```

### Show Commit Details

```bash
# Show commit details
git show <commit>

# Show specific file at commit
git show <commit>:<file>

# Show commit statistics
git show --stat <commit>

# Show commit names only
git show --name-only <commit>

# Show commit with word diff
git show --word-diff <commit>
```

### Diff

```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --staged
git diff --cached

# Show changes in specific file
git diff <file>

# Compare branches
git diff <branch1>..<branch2>

# Compare commits
git diff <commit1> <commit2>

# Compare with specific commit
git diff HEAD~1

# Word-level diff
git diff --word-diff

# Show statistics only
git diff --stat

# Show file names only
git diff --name-only

# Show file names with status
git diff --name-status

# Ignore whitespace
git diff -w
git diff --ignore-all-space
```

### Blame

```bash
# Show who changed each line
git blame <file>

# Show blame for specific lines
git blame -L 10,20 <file>

# Show blame with email
git blame -e <file>

# Ignore whitespace changes
git blame -w <file>
```

### Reflog

```bash
# Show reference log (command history)
git reflog

# Show reflog for specific branch
git reflog <branch>

# Show reflog with dates
git reflog --date=relative

# Expire old reflog entries
git reflog expire --expire=30.days.ago --all
```

## Branching

### Create and Switch Branches

```bash
# List branches
git branch

# List all branches (including remote)
git branch -a

# List remote branches
git branch -r

# Create new branch
git branch <branch_name>

# Create branch from specific commit
git branch <branch_name> <commit>

# Switch to branch
git checkout <branch_name>

# Create and switch to new branch
git checkout -b <branch_name>

# Create branch from specific commit and switch
git checkout -b <branch_name> <commit>

# Modern syntax (Git 2.23+)
git switch <branch_name>
git switch -c <branch_name>

# Switch to previous branch
git checkout -
git switch -
```

### Delete Branches

```bash
# Delete local branch (safe)
git branch -d <branch_name>

# Force delete local branch
git branch -D <branch_name>

# Delete remote branch
git push origin --delete <branch_name>
git push origin :<branch_name>
```

### Rename Branches

```bash
# Rename current branch
git branch -m <new_name>

# Rename specific branch
git branch -m <old_name> <new_name>

# Rename and push to remote
git branch -m <new_name>
git push origin -u <new_name>
git push origin --delete <old_name>
```

### Branch Information

```bash
# Show branches with last commit
git branch -v

# Show merged branches
git branch --merged

# Show unmerged branches
git branch --no-merged

# Show branches containing commit
git branch --contains <commit>

# Track remote branch
git branch --set-upstream-to=origin/<branch>
git branch -u origin/<branch>
```

## Merging

### Basic Merge

```bash
# Merge branch into current branch
git merge <branch>

# Merge with commit message
git merge <branch> -m "Merge message"

# Merge without fast-forward
git merge --no-ff <branch>

# Merge with fast-forward only
git merge --ff-only <branch>

# Squash merge (combine all commits)
git merge --squash <branch>

# Abort merge
git merge --abort

# Continue merge after resolving conflicts
git merge --continue
```

### Merge Strategies

```bash
# Use recursive strategy (default)
git merge -s recursive <branch>

# Use ours strategy (keep our version)
git merge -s ours <branch>

# Use theirs strategy
git merge -X theirs <branch>

# Ignore whitespace during merge
git merge -X ignore-all-space <branch>
```

## Rebasing

### Basic Rebase

```bash
# Rebase current branch onto another
git rebase <branch>

# Rebase onto specific commit
git rebase <commit>

# Continue rebase after resolving conflicts
git rebase --continue

# Skip current commit
git rebase --skip

# Abort rebase
git rebase --abort

# Rebase and preserve merges
git rebase -p <branch>
```

### Interactive Rebase

```bash
# Interactive rebase last N commits
git rebase -i HEAD~3

# Interactive rebase from specific commit
git rebase -i <commit>

# Interactive rebase with autosquash
git rebase -i --autosquash <branch>

# Commands in interactive rebase:
# pick (p)   = use commit
# reword (r) = use commit, but edit message
# edit (e)   = use commit, but stop for amending
# squash (s) = merge with previous commit
# fixup (f)  = like squash, but discard message
# drop (d)   = remove commit
# exec (x)   = run shell command
```

## Remote Operations

### Remote Management

```bash
# List remotes
git remote

# List remotes with URLs
git remote -v

# Add remote
git remote add <name> <url>

# Remove remote
git remote remove <name>
git remote rm <name>

# Rename remote
git remote rename <old> <new>

# Change remote URL
git remote set-url <name> <new_url>

# Show remote details
git remote show <name>

# Prune stale remote branches
git remote prune origin
```

### Fetch

```bash
# Fetch from remote
git fetch

# Fetch from specific remote
git fetch <remote>

# Fetch specific branch
git fetch <remote> <branch>

# Fetch all remotes
git fetch --all

# Fetch and prune deleted remote branches
git fetch -p
git fetch --prune

# Fetch tags
git fetch --tags

# Dry run (show what would be fetched)
git fetch --dry-run
```

### Pull

```bash
# Pull from tracked remote branch
git pull

# Pull from specific remote and branch
git pull <remote> <branch>

# Pull with rebase
git pull --rebase

# Pull with fast-forward only
git pull --ff-only

# Pull all submodules
git pull --recurse-submodules

# Pull and prune
git pull -p
```

### Push

```bash
# Push to remote
git push

# Push to specific remote and branch
git push <remote> <branch>

# Push and set upstream
git push -u <remote> <branch>

# Push all branches
git push --all

# Push tags
git push --tags

# Push specific tag
git push <remote> <tag>

# Force push (dangerous!)
git push --force

# Safer force push (checks remote state)
git push --force-with-lease

# Delete remote branch
git push <remote> --delete <branch>

# Delete remote tag
git push <remote> --delete <tag>

# Dry run (show what would be pushed)
git push --dry-run
```

## Undoing Changes

### Working Directory

```bash
# Discard changes in file
git checkout -- <file>

# Discard all changes
git checkout -- .

# Modern syntax
git restore <file>
git restore .

# Restore from specific commit
git restore --source=<commit> <file>

# Clean untracked files
git clean -f

# Clean untracked files and directories
git clean -fd

# Clean ignored files too
git clean -fdx

# Dry run (show what would be removed)
git clean -n
```

### Staging Area

```bash
# Unstage file
git reset HEAD <file>

# Unstage all files
git reset HEAD

# Modern syntax
git restore --staged <file>
git restore --staged .
```

### Commits

```bash
# Undo last commit, keep changes staged
git reset --soft HEAD~1

# Undo last commit, keep changes unstaged
git reset --mixed HEAD~1
git reset HEAD~1

# Undo last commit, discard changes
git reset --hard HEAD~1

# Reset to specific commit
git reset --hard <commit>

# Create new commit that undoes changes
git revert <commit>

# Revert merge commit
git revert -m 1 <merge_commit>

# Revert without committing
git revert -n <commit>

# Revert range of commits
git revert <commit1>..<commit2>
```

## Stashing

### Basic Stash

```bash
# Stash current changes
git stash

# Stash with message
git stash save "Work in progress"
git stash push -m "Work in progress"

# Stash including untracked files
git stash -u
git stash --include-untracked

# Stash including ignored files
git stash -a
git stash --all

# Stash specific files
git stash push <file>

# Stash with patch mode
git stash -p
```

### Managing Stashes

```bash
# List stashes
git stash list

# Show stash contents
git stash show
git stash show -p

# Show specific stash
git stash show stash@{1}

# Apply last stash
git stash apply

# Apply specific stash
git stash apply stash@{1}

# Apply and remove stash (pop)
git stash pop

# Pop specific stash
git stash pop stash@{1}

# Create branch from stash
git stash branch <branch_name>

# Drop specific stash
git stash drop stash@{1}

# Clear all stashes
git stash clear
```

## Tags

### Create Tags

```bash
# List tags
git tag

# List tags with pattern
git tag -l "v1.*"

# Create lightweight tag
git tag <tag_name>

# Create annotated tag
git tag -a <tag_name> -m "Tag message"

# Tag specific commit
git tag <tag_name> <commit>

# Tag with specific date
git tag -a <tag_name> -m "Message" --date="2024-01-01"
```

### Manage Tags

```bash
# Show tag details
git show <tag_name>

# Delete local tag
git tag -d <tag_name>

# Delete remote tag
git push origin --delete <tag_name>
git push origin :refs/tags/<tag_name>

# Push tag to remote
git push origin <tag_name>

# Push all tags
git push --tags

# Fetch tags from remote
git fetch --tags

# Checkout tag (creates detached HEAD)
git checkout <tag_name>

# Create branch from tag
git checkout -b <branch_name> <tag_name>
```

## Advanced Operations

### Cherry-Pick

```bash
# Apply specific commit
git cherry-pick <commit>

# Apply multiple commits
git cherry-pick <commit1> <commit2>

# Apply commit range
git cherry-pick <commit1>..<commit2>

# Cherry-pick without committing
git cherry-pick -n <commit>

# Continue cherry-pick
git cherry-pick --continue

# Abort cherry-pick
git cherry-pick --abort

# Skip current commit
git cherry-pick --skip
```

### Bisect

```bash
# Start bisect
git bisect start

# Mark current commit as bad
git bisect bad

# Mark current commit as good
git bisect good

# Mark specific commit as good
git bisect good <commit>

# Skip current commit
git bisect skip

# Reset bisect
git bisect reset

# Visualize bisect
git bisect visualize

# Run automated bisect
git bisect run <script>
```

### Submodules

```bash
# Add submodule
git submodule add <repository_url> <path>

# Initialize submodules
git submodule init

# Update submodules
git submodule update

# Clone with submodules
git clone --recursive <repository_url>

# Update all submodules
git submodule update --remote

# Remove submodule
git submodule deinit <path>
git rm <path>

# Show submodule status
git submodule status

# Foreach command on all submodules
git submodule foreach <command>
```

### Worktrees

```bash
# List worktrees
git worktree list

# Add new worktree
git worktree add <path> <branch>

# Add worktree with new branch
git worktree add -b <new_branch> <path>

# Remove worktree
git worktree remove <path>

# Prune worktree information
git worktree prune
```

### Archive

```bash
# Create archive of repository
git archive --format=zip HEAD > archive.zip

# Archive specific branch
git archive --format=tar <branch> > archive.tar

# Archive with prefix
git archive --prefix=project/ HEAD > archive.tar

# Archive specific directory
git archive HEAD <directory>/ > archive.tar
```

## Maintenance

### Repository Maintenance

```bash
# Run garbage collection
git gc

# Aggressive garbage collection
git gc --aggressive

# Prune unreachable objects
git prune

# Verify repository integrity
git fsck

# Show repository statistics
git count-objects -v

# Show repository size
git count-objects -vH
```

### Optimization

```bash
# Repack repository
git repack

# Aggressive repack
git repack -a -d --depth=250 --window=250

# Prune old reflog entries
git reflog expire --expire=30.days.ago --all

# Remove old objects
git prune --expire=30.days.ago
```

## Searching

### Grep

```bash
# Search for text in repository
git grep "pattern"

# Search with line numbers
git grep -n "pattern"

# Search for whole word
git grep -w "pattern"

# Search case-insensitively
git grep -i "pattern"

# Search in specific commit
git grep "pattern" <commit>

# Search with context
git grep -C 2 "pattern"

# Show file names only
git grep -l "pattern"

# Count matches per file
git grep -c "pattern"

# Search with AND condition
git grep -e "pattern1" --and -e "pattern2"

# Search with OR condition
git grep -e "pattern1" --or -e "pattern2"
```

### Log Search

```bash
# Search commit messages
git log --grep="pattern"

# Search commit content
git log -S "code"

# Search with pickaxe (show diff)
git log -G "regex"

# Search author
git log --author="name"

# Search committer
git log --committer="name"
```

## Help

```bash
# Show help for command
git help <command>
git <command> --help

# Show quick help
git <command> -h

# Show all commands
git help -a

# Show guides
git help -g

# Show config options
git help config
```