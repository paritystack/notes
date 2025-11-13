# Git Cheatsheet

Quick reference for the most commonly used Git commands.

## Setup and Configuration

```bash
# Initial setup
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "vim"
git config --global init.defaultBranch main

# View configuration
git config --list
git config user.name
```

## Repository Creation

```bash
# Create new repository
git init

# Clone existing repository
git clone <url>
git clone <url> <directory>
git clone -b <branch> <url>
```

## Daily Workflow

```bash
# Check status
git status
git status -s                # Short format

# Add files
git add <file>
git add .                    # Add all
git add -p                   # Interactive staging

# Commit changes
git commit -m "Message"
git commit -am "Message"     # Stage and commit
git commit --amend           # Modify last commit

# View history
git log
git log --oneline
git log --graph --oneline --all

# Push/Pull
git pull
git pull --rebase
git push
git push -u origin <branch>
```

## Branching

```bash
# List branches
git branch                   # Local branches
git branch -a                # All branches
git branch -r                # Remote branches

# Create branch
git branch <name>
git checkout -b <name>       # Create and switch
git switch -c <name>         # Modern syntax

# Switch branches
git checkout <name>
git switch <name>
git checkout -               # Previous branch

# Delete branch
git branch -d <name>         # Safe delete
git branch -D <name>         # Force delete
git push origin --delete <name>  # Delete remote

# Rename branch
git branch -m <new_name>
```

## Merging and Rebasing

```bash
# Merge
git merge <branch>
git merge --no-ff <branch>
git merge --squash <branch>
git merge --abort

# Rebase
git rebase <branch>
git rebase -i HEAD~3         # Interactive rebase
git rebase --continue
git rebase --abort

# Resolve conflicts
git status                   # Check conflicts
# Edit files to resolve
git add <resolved_file>
git commit                   # or git rebase --continue
```

## Remote Operations

```bash
# List remotes
git remote -v

# Add/Remove remote
git remote add origin <url>
git remote remove <name>
git remote set-url origin <new_url>

# Fetch
git fetch
git fetch origin
git fetch --all
git fetch -p                 # Prune deleted branches

# Pull
git pull
git pull origin <branch>
git pull --rebase

# Push
git push
git push origin <branch>
git push -u origin <branch>
git push --tags
git push --force-with-lease  # Safer force push
```

## Viewing Changes

```bash
# Show changes
git diff                     # Unstaged changes
git diff --staged            # Staged changes
git diff <branch1> <branch2>
git diff HEAD~1

# Show commits
git log
git log --oneline
git log -p                   # With patches
git log --stat               # With statistics
git log --author="Name"
git log --since="2 weeks ago"
git log --grep="pattern"

# Show commit details
git show <commit>
git show <commit>:<file>

# Blame
git blame <file>
```

## Undoing Changes

```bash
# Discard changes in working directory
git restore <file>
git restore .
git checkout -- <file>       # Old syntax

# Unstage files
git restore --staged <file>
git reset HEAD <file>        # Old syntax

# Undo commits
git reset --soft HEAD~1      # Keep changes staged
git reset HEAD~1             # Keep changes unstaged
git reset --hard HEAD~1      # Discard changes
git revert <commit>          # Create reverting commit

# Clean untracked files
git clean -n                 # Dry run
git clean -f                 # Remove files
git clean -fd                # Remove files and directories
```

## Stashing

```bash
# Save changes temporarily
git stash
git stash save "Message"
git stash -u                 # Include untracked

# List stashes
git stash list

# Apply stash
git stash apply
git stash apply stash@{2}
git stash pop                # Apply and remove

# Manage stashes
git stash show -p
git stash drop stash@{0}
git stash clear
```

## Tags

```bash
# List tags
git tag
git tag -l "v1.*"

# Create tags
git tag <name>               # Lightweight
git tag -a <name> -m "Message"  # Annotated

# Push tags
git push origin <tag>
git push --tags

# Delete tags
git tag -d <name>
git push origin --delete <name>

# Checkout tag
git checkout <tag>
```

## Advanced Commands

```bash
# Cherry-pick
git cherry-pick <commit>
git cherry-pick <commit1> <commit2>

# Bisect (find bug)
git bisect start
git bisect bad
git bisect good <commit>
# Test and mark good/bad
git bisect reset

# Reflog (recover lost commits)
git reflog
git checkout <commit>

# Archive
git archive --format=zip HEAD > archive.zip

# Search
git grep "pattern"
git grep -n "pattern"        # With line numbers
git log -S "code"            # Commits with code
```

## Collaboration Workflows

### Feature Branch Workflow

```bash
# Start new feature
git checkout -b feature/<name>

# Work on feature
git add .
git commit -m "Add feature"

# Push feature branch
git push -u origin feature/<name>

# After PR is merged
git checkout main
git pull
git branch -d feature/<name>
```

### Sync with Upstream

```bash
# Fork workflow
git remote add upstream <original_repo_url>
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Hotfix Workflow

```bash
# Create hotfix from main
git checkout main
git checkout -b hotfix/<issue>

# Fix and commit
git add .
git commit -m "Fix issue"

# Merge to main and develop
git checkout main
git merge --no-ff hotfix/<issue>
git tag -a v1.0.1 -m "Version 1.0.1"

git checkout develop
git merge --no-ff hotfix/<issue>

# Cleanup
git branch -d hotfix/<issue>
```

## Common Scenarios

### Forgot to create branch

```bash
git stash
git checkout -b feature/<name>
git stash pop
```

### Undo last commit but keep changes

```bash
git reset --soft HEAD~1
```

### Amend commit message

```bash
git commit --amend -m "New message"
```

### Remove file from Git but keep locally

```bash
git rm --cached <file>
```

### Sync fork with original

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Squash last N commits

```bash
git rebase -i HEAD~N
# Change "pick" to "squash" for commits to combine
```

### Change author of last commit

```bash
git commit --amend --author="Name <email>"
```

### Create orphan branch

```bash
git switch --orphan <branch>
git commit --allow-empty -m "Initial commit"
git push -u origin <branch>
```

## Configuration Aliases

```bash
# Create useful aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.lg 'log --graph --oneline --all'
git config --global alias.amend 'commit --amend --no-edit'
```

## .gitignore Patterns

```bash
# Ignore files
*.log
*.tmp
.env

# Ignore directories
node_modules/
dist/
build/

# Ignore with exceptions
*.a
!lib.a

# IDE files
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db
```

## Common Options

```bash
# Flags used with multiple commands
-a, --all               # All
-b, --branch            # Branch
-d, --delete            # Delete
-f, --force             # Force
-m, --message           # Message
-n, --dry-run           # Dry run
-p, --patch             # Interactive patch mode
-u, --set-upstream      # Set upstream
-v, --verbose           # Verbose output

# Common patterns
HEAD                    # Current commit
HEAD~1                  # Previous commit
HEAD~n                  # N commits ago
HEAD^                   # First parent of merge
<commit>                # Commit hash
<branch>                # Branch name
origin                  # Default remote name
main/master             # Default branch names
```

## Emergency Commands

```bash
# Abort everything
git merge --abort
git rebase --abort
git cherry-pick --abort

# Recover lost work
git reflog
git checkout <lost_commit>
git branch recover-branch <lost_commit>

# Undo force push (if possible)
git reflog
git reset --hard <previous_commit>
git push --force-with-lease

# Remove sensitive data from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch <file>" \
  --prune-empty --tag-name-filter cat -- --all
```

## Quick Reference: Git States

```
Working Directory → Staging Area → Repository → Remote
      (edit)          (add)        (commit)    (push)
```

## Quick Reference: Undoing

```
Working Directory:    git restore <file>
Staging Area:         git restore --staged <file>
Last Commit:          git commit --amend
Previous Commits:     git revert <commit>
Local Branch:         git reset --hard <commit>
```

## Quick Reference: Branch Management

```
Create:    git checkout -b <name>
Switch:    git switch <name>
Merge:     git merge <name>
Delete:    git branch -d <name>
Remote:    git push -u origin <name>
```
