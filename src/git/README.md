# Git Version Control

## Overview

Git is a distributed version control system designed to handle everything from small to very large projects with speed and efficiency. Created by Linus Torvalds in 2005, Git has become the de facto standard for version control in software development.

## What is Version Control?

Version control is a system that records changes to files over time so that you can recall specific versions later. It allows you to:

- Track changes to your code
- Collaborate with other developers
- Revert to previous versions
- Create branches for experimental features
- Merge changes from multiple sources
- Maintain a complete history of your project

## Why Git?

1. **Distributed**: Every developer has a full copy of the repository
2. **Fast**: Most operations are local
3. **Branching**: Lightweight and powerful branching model
4. **Data Integrity**: Cryptographic hash (SHA-1) ensures data integrity
5. **Staging Area**: Review changes before committing
6. **Open Source**: Free and widely supported

## Git Basics

### The Three States

Git has three main states for your files:

1. **Modified**: Changed but not committed
2. **Staged**: Marked for next commit
3. **Committed**: Safely stored in local database

```
Working Directory -> Staging Area -> Git Repository
     (edit)            (stage)         (commit)
```

### Git Workflow

```bash
# 1. Make changes in working directory
echo "Hello World" > file.txt

# 2. Stage changes
git add file.txt

# 3. Commit changes
git commit -m "Add hello world file"

# 4. Push to remote repository
git push origin main
```

## Installation and Setup

### Installation

```bash
# Linux (Debian/Ubuntu)
sudo apt-get update
sudo apt-get install git

# Linux (Fedora)
sudo dnf install git

# macOS (Homebrew)
brew install git

# Windows
# Download from https://git-scm.com/download/win

# Verify installation
git --version
```

### Initial Configuration

```bash
# Set user name
git config --global user.name "Your Name"

# Set email
git config --global user.email "your.email@example.com"

# Set default editor
git config --global core.editor "vim"

# Set default branch name
git config --global init.defaultBranch main

# Enable color output
git config --global color.ui auto

# View all settings
git config --list

# View specific setting
git config user.name

# Edit config file directly
git config --global --edit
```

## Basic Commands

### Creating Repositories

```bash
# Initialize new repository
git init

# Clone existing repository
git clone https://github.com/user/repo.git

# Clone to specific directory
git clone https://github.com/user/repo.git my-project

# Clone specific branch
git clone -b develop https://github.com/user/repo.git
```

### Making Changes

```bash
# Check status
git status

# Add file to staging
git add file.txt

# Add all files
git add .

# Add all files with specific extension
git add *.js

# Interactive staging
git add -p

# Commit staged changes
git commit -m "Commit message"

# Commit with detailed message
git commit

# Stage and commit in one step
git commit -am "Message"

# Amend last commit
git commit --amend

# Amend without changing message
git commit --amend --no-edit
```

### Viewing History

```bash
# View commit history
git log

# Compact log
git log --oneline

# Graph view
git log --graph --oneline --all

# Limit number of commits
git log -n 5

# Show commits by author
git log --author="John"

# Show commits in date range
git log --since="2 weeks ago"
git log --until="2024-01-01"

# Show file changes
git log --stat

# Show detailed changes
git log -p

# Search commit messages
git log --grep="fix"

# Show commits affecting specific file
git log -- file.txt
```

### Viewing Changes

```bash
# Show unstaged changes
git diff

# Show staged changes
git diff --staged

# Show changes in specific file
git diff file.txt

# Compare branches
git diff main..feature

# Compare commits
git diff commit1 commit2

# Word-level diff
git diff --word-diff
```

## Branching and Merging

### Branches

```bash
# List branches
git branch

# List all branches (including remote)
git branch -a

# Create new branch
git branch feature-name

# Switch to branch
git checkout feature-name

# Create and switch in one command
git checkout -b feature-name

# Modern syntax (Git 2.23+)
git switch feature-name
git switch -c feature-name

# Delete branch
git branch -d feature-name

# Force delete unmerged branch
git branch -D feature-name

# Rename current branch
git branch -m new-name

# Rename specific branch
git branch -m old-name new-name
```

### Merging

```bash
# Merge branch into current branch
git merge feature-name

# Merge with commit message
git merge feature-name -m "Merge feature"

# Merge without fast-forward
git merge --no-ff feature-name

# Abort merge
git merge --abort

# Continue merge after resolving conflicts
git merge --continue
```

### Handling Merge Conflicts

```bash
# When merge conflict occurs:

# 1. Check conflicted files
git status

# 2. Open files and resolve conflicts
#    Look for markers: <<<<<<<, =======, >>>>>>>

# 3. After resolving, stage files
git add resolved-file.txt

# 4. Complete merge
git commit

# Or use merge tool
git mergetool
```

### Rebasing

```bash
# Rebase current branch onto main
git rebase main

# Interactive rebase (last 3 commits)
git rebase -i HEAD~3

# Continue after resolving conflicts
git rebase --continue

# Skip current commit
git rebase --skip

# Abort rebase
git rebase --abort

# Rebase options in interactive mode:
# pick   = use commit
# reword = use commit, but edit message
# edit   = use commit, but stop for amending
# squash = merge with previous commit
# drop   = remove commit
```

## Remote Repositories

### Working with Remotes

```bash
# List remotes
git remote

# List remotes with URLs
git remote -v

# Add remote
git remote add origin https://github.com/user/repo.git

# Change remote URL
git remote set-url origin https://github.com/user/new-repo.git

# Remove remote
git remote remove origin

# Rename remote
git remote rename origin upstream

# Show remote info
git remote show origin
```

### Fetching and Pulling

```bash
# Fetch from remote (doesn't merge)
git fetch origin

# Fetch all remotes
git fetch --all

# Pull (fetch + merge)
git pull origin main

# Pull with rebase
git pull --rebase origin main

# Pull specific branch
git pull origin feature-branch
```

### Pushing

```bash
# Push to remote
git push origin main

# Push and set upstream
git push -u origin main

# Push all branches
git push --all origin

# Push tags
git push --tags

# Force push (dangerous!)
git push --force origin main

# Safer force push
git push --force-with-lease origin main

# Delete remote branch
git push origin --delete branch-name
```

## Undoing Changes

### Working Directory

```bash
# Discard changes in file
git checkout -- file.txt

# Discard all changes
git checkout -- .

# Modern syntax
git restore file.txt
git restore .

# Remove untracked files
git clean -f

# Remove untracked files and directories
git clean -fd

# Preview what will be removed
git clean -n
```

### Staging Area

```bash
# Unstage file
git reset HEAD file.txt

# Unstage all files
git reset HEAD

# Modern syntax
git restore --staged file.txt
```

### Commits

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Undo multiple commits
git reset --hard HEAD~3

# Reset to specific commit
git reset --hard commit-hash

# Create new commit that undoes changes
git revert commit-hash

# Revert multiple commits
git revert commit1..commit3
```

## Advanced Features

### Stashing

```bash
# Stash current changes
git stash

# Stash with message
git stash save "Work in progress"

# List stashes
git stash list

# Apply last stash
git stash apply

# Apply specific stash
git stash apply stash@{2}

# Apply and remove stash
git stash pop

# Create branch from stash
git stash branch feature-name

# Drop stash
git stash drop stash@{0}

# Clear all stashes
git stash clear

# Stash including untracked files
git stash -u
```

### Tags

```bash
# List tags
git tag

# Create lightweight tag
git tag v1.0.0

# Create annotated tag
git tag -a v1.0.0 -m "Version 1.0.0"

# Tag specific commit
git tag v1.0.0 commit-hash

# Push tag to remote
git push origin v1.0.0

# Push all tags
git push --tags

# Delete local tag
git tag -d v1.0.0

# Delete remote tag
git push origin --delete v1.0.0

# Checkout tag
git checkout v1.0.0
```

### Cherry-Pick

```bash
# Apply specific commit to current branch
git cherry-pick commit-hash

# Cherry-pick multiple commits
git cherry-pick commit1 commit2

# Cherry-pick without committing
git cherry-pick -n commit-hash

# Abort cherry-pick
git cherry-pick --abort
```

### Bisect

```bash
# Start bisect session
git bisect start

# Mark current commit as bad
git bisect bad

# Mark known good commit
git bisect good commit-hash

# Git will checkout middle commit
# Test and mark as good or bad
git bisect good  # or git bisect bad

# Continue until bug is found

# End bisect session
git bisect reset
```

## Git Workflows

### Feature Branch Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes and commit
git add .
git commit -m "Implement new feature"

# 3. Push to remote
git push -u origin feature/new-feature

# 4. Create pull request (on GitHub/GitLab)

# 5. After review, merge via web interface

# 6. Update local main branch
git checkout main
git pull origin main

# 7. Delete feature branch
git branch -d feature/new-feature
git push origin --delete feature/new-feature
```

### Gitflow Workflow

```bash
# Main branches: main (production), develop (integration)

# Start new feature
git checkout -b feature/feature-name develop

# Finish feature
git checkout develop
git merge --no-ff feature/feature-name
git branch -d feature/feature-name

# Start release
git checkout -b release/1.0.0 develop

# Finish release
git checkout main
git merge --no-ff release/1.0.0
git tag -a v1.0.0
git checkout develop
git merge --no-ff release/1.0.0
git branch -d release/1.0.0

# Hotfix
git checkout -b hotfix/fix-bug main
git checkout main
git merge --no-ff hotfix/fix-bug
git tag -a v1.0.1
git checkout develop
git merge --no-ff hotfix/fix-bug
git branch -d hotfix/fix-bug
```

### Fork and Pull Request Workflow

```bash
# 1. Fork repository on GitHub

# 2. Clone your fork
git clone https://github.com/your-username/repo.git
cd repo

# 3. Add upstream remote
git remote add upstream https://github.com/original-owner/repo.git
git remote -v

# 4. Create feature branch
git checkout -b feature/my-feature

# 5. Make changes and commit
git add .
git commit -m "Add new feature"

# 6. Keep your fork updated
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# 7. Rebase your feature branch (optional but recommended)
git checkout feature/my-feature
git rebase main

# 8. Push to your fork
git push origin feature/my-feature

# 9. Create Pull Request on GitHub
#    - Navigate to original repository
#    - Click "New Pull Request"
#    - Select your fork and branch

# 10. After PR is merged, update and cleanup
git checkout main
git pull upstream main
git push origin main
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```

### Trunk-Based Development

```bash
# Work directly on main branch with short-lived feature branches

# 1. Create short-lived feature branch
git checkout -b feature/quick-fix

# 2. Make small, incremental changes
git add .
git commit -m "Implement part 1 of feature"

# 3. Keep branch up to date with main (multiple times per day)
git checkout main
git pull origin main
git checkout feature/quick-fix
git rebase main

# 4. Merge back to main quickly (within hours or 1-2 days)
git checkout main
git merge --no-ff feature/quick-fix
git push origin main

# 5. Delete feature branch
git branch -d feature/quick-fix

# Alternative: Direct commits to main (for very small changes)
git checkout main
git pull origin main
# Make small change
git add .
git commit -m "Fix typo"
git push origin main
```

### Release Branch Workflow

```bash
# Create release branch from main
git checkout -b release/v2.0.0 main

# Make release-specific changes (version bumps, changelog, etc.)
git add .
git commit -m "Prepare release v2.0.0"

# Test the release branch thoroughly
# Fix any bugs found
git add .
git commit -m "Fix release bug"

# Merge to main and tag
git checkout main
git merge --no-ff release/v2.0.0
git tag -a v2.0.0 -m "Release version 2.0.0"
git push origin main
git push origin v2.0.0

# Merge release changes back to develop (if using Gitflow)
git checkout develop
git merge --no-ff release/v2.0.0

# Delete release branch
git branch -d release/v2.0.0
```

## Daily Workflow Patterns

### Start of Day

```bash
# Update your local repository
git checkout main
git pull origin main

# Check what you were working on
git status
git log --oneline -5

# Resume work on feature branch
git checkout feature/my-feature
git rebase main
```

### During Development

```bash
# Check status frequently
git status

# View changes before staging
git diff

# Stage changes selectively
git add -p  # Interactive staging

# Commit with meaningful message
git commit -m "feat: Add user authentication

Implement JWT-based authentication system with:
- Login endpoint
- Token validation middleware
- Logout functionality

Refs #123"

# Push to remote frequently
git push origin feature/my-feature

# Save work in progress without committing
git stash save "WIP: working on login form"
```

### Before Creating Pull Request

```bash
# Make sure branch is up to date
git checkout main
git pull origin main
git checkout feature/my-feature
git rebase main

# Clean up commit history (if needed)
git rebase -i HEAD~5
# Squash, reword, or reorder commits

# Run tests
npm test  # or your test command

# Push updated branch
git push --force-with-lease origin feature/my-feature

# Create Pull Request on GitHub
```

### After Pull Request Review

```bash
# Address review comments
git add .
git commit -m "Address PR feedback"

# Or amend last commit
git add .
git commit --amend --no-edit

# Force push (your PR branch)
git push --force-with-lease origin feature/my-feature
```

### End of Day

```bash
# Commit work in progress
git add .
git commit -m "WIP: partial implementation"

# Or stash if not ready to commit
git stash save "WIP: end of day $(date)"

# Push to remote as backup
git push origin feature/my-feature
```

### Working with Multiple Features

```bash
# Save current work
git stash

# Switch to different feature
git checkout feature/other-feature

# Work on it...
git add .
git commit -m "Update feature"

# Switch back to original feature
git checkout feature/my-feature
git stash pop
```

## Common Workflow Scenarios

### Fixing a Bug in Production

```bash
# 1. Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-bug

# 2. Fix the bug
git add .
git commit -m "fix: Resolve critical authentication bug

Fix issue where users couldn't login after password reset.

Fixes #456"

# 3. Test thoroughly
npm test

# 4. Merge to main
git checkout main
git merge --no-ff hotfix/critical-bug
git tag -a v1.0.1 -m "Hotfix release 1.0.1"

# 5. Push to production
git push origin main
git push origin v1.0.1

# 6. Merge back to develop
git checkout develop
git merge --no-ff hotfix/critical-bug

# 7. Cleanup
git branch -d hotfix/critical-bug
```

### Syncing Fork with Upstream

```bash
# Add upstream if not already added
git remote add upstream https://github.com/original/repo.git

# Fetch upstream changes
git fetch upstream

# Merge upstream changes to main
git checkout main
git merge upstream/main

# Push to your fork
git push origin main

# Update your feature branch
git checkout feature/my-feature
git rebase main
```

### Collaborating on a Branch

```bash
# Person A creates branch and pushes
git checkout -b feature/shared-feature
git add .
git commit -m "Initial implementation"
git push -u origin feature/shared-feature

# Person B clones and contributes
git fetch origin
git checkout feature/shared-feature
git add .
git commit -m "Add tests"
git push origin feature/shared-feature

# Person A pulls updates
git checkout feature/shared-feature
git pull origin feature/shared-feature
```

### Recovering from Mistakes

```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Discard all local changes
git reset --hard HEAD

# Recover deleted branch
git reflog
git checkout -b recovered-branch <commit-hash>

# Undo force push (if reflog available)
git reflog
git reset --hard HEAD@{n}
git push --force-with-lease

# Revert a merged PR
git revert -m 1 <merge-commit-hash>
git push origin main
```

### Working with Large Files

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.psd"
git lfs track "*.mp4"
git lfs track "datasets/*"

# Add .gitattributes
git add .gitattributes

# Add and commit large files
git add large-file.psd
git commit -m "Add design file"
git push origin main
```

### Maintaining Clean History

```bash
# Squash commits before merging
git checkout feature/my-feature
git rebase -i main

# In editor, change "pick" to "squash" for commits to combine

# Rewrite commit message
git commit --amend

# Force push (only on feature branches!)
git push --force-with-lease origin feature/my-feature
```

## Best Practices

### Commit Messages

```bash
# Good commit message structure:
# <type>: <subject>
# 
# <body>
# 
# <footer>

# Example:
git commit -m "feat: Add user authentication

Implement JWT-based authentication system with login and logout endpoints.
Uses bcrypt for password hashing.

Closes #123"

# Common types:
# feat:     New feature
# fix:      Bug fix
# docs:     Documentation changes
# style:    Formatting, missing semicolons, etc.
# refactor: Code restructuring
# test:     Adding tests
# chore:    Maintenance tasks
```

### General Best Practices

1. **Commit Often**: Make small, logical commits
2. **Write Clear Messages**: Explain what and why
3. **Use Branches**: Keep main stable
4. **Pull Before Push**: Stay synchronized
5. **Review Before Commit**: Check what you're committing
6. **Don't Commit Secrets**: Use .gitignore for sensitive files
7. **Keep History Clean**: Use rebase for feature branches
8. **Tag Releases**: Mark important versions
9. **Backup Remote**: Always have a remote backup
10. **Learn to Revert**: Know how to undo mistakes

### .gitignore

```bash
# Create .gitignore file
cat > .gitignore << 'EOL'
# Dependencies
node_modules/
vendor/

# Environment files
.env
.env.local

# Build outputs
dist/
build/
*.log

# IDE files
.vscode/
.idea/
*.swp

# OS files
.DS_Store
Thumbs.db

# Compiled files
*.pyc
*.class
*.o
EOL

# Global gitignore
git config --global core.excludesfile ~/.gitignore_global
```

## Troubleshooting

### Common Issues

```bash
# Undo last commit but keep changes
git reset --soft HEAD~1

# Fix wrong commit message
git commit --amend -m "Correct message"

# Recover deleted branch
git reflog
git checkout -b recovered-branch commit-hash

# Resolve "detached HEAD"
git checkout main

# Remove file from Git but keep locally
git rm --cached file.txt

# Update .gitignore for already tracked files
git rm -r --cached .
git add .
git commit -m "Update .gitignore"

# Find commit that introduced bug
git bisect start
git bisect bad
git bisect good commit-hash
```

### Performance

```bash
# Clean up repository
git gc

# Aggressive cleanup
git gc --aggressive

# Prune unreachable objects
git prune

# Show repository size
git count-objects -vH
```

## Git Aliases

```bash
# Create useful aliases
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --graph --oneline --all'
git config --global alias.amend 'commit --amend --no-edit'

# Use aliases
git co main
git ci -m "Message"
git visual
```

## Integration with GitHub

GitHub adds collaboration features on top of Git. See the dedicated [GitHub guide](github.md) for:

- Pull requests
- Issues
- Actions (CI/CD)
- Pages
- Wikis
- Organizations and teams

## Available Resources

- [Git Cheat Sheet](cheatsheet.md) - Quick reference guide
- [Git Commands](commands.md) - Comprehensive command list
- [GitHub Guide](github.md) - GitHub-specific features

## Learning Resources

### Documentation
- [Official Git Documentation](https://git-scm.com/doc)
- [Pro Git Book](https://git-scm.com/book/en/v2) (free online)
- [Git Reference](https://git-scm.com/docs)

### Interactive Tutorials
- [Learn Git Branching](https://learngitbranching.js.org/)
- [GitHub Learning Lab](https://lab.github.com/)
- [Git Exercises](https://gitexercises.fracz.com/)

### Visualizations
- [Visualizing Git Concepts](https://git-school.github.io/visualizing-git/)
- [Git Flow Diagram](https://nvie.com/posts/a-successful-git-branching-model/)

## Quick Reference

### Daily Commands
```bash
git status                    # Check status
git add .                     # Stage all changes
git commit -m "message"       # Commit changes
git pull                      # Update from remote
git push                      # Push to remote
git log --oneline            # View history
```

### Branching
```bash
git branch                    # List branches
git checkout -b feature       # Create and switch
git merge feature            # Merge branch
git branch -d feature        # Delete branch
```

### Undoing
```bash
git reset --soft HEAD~1      # Undo commit, keep changes
git restore file.txt         # Discard file changes
git revert commit-hash       # Create revert commit
git stash                    # Save temporary changes
```

### Remote
```bash
git remote -v                # List remotes
git fetch origin             # Download changes
git pull origin main         # Fetch and merge
git push origin main         # Upload changes
```

## Next Steps

1. Practice basic commands: add, commit, push, pull
2. Learn branching and merging
3. Master undoing changes safely
4. Explore advanced features: rebase, cherry-pick, bisect
5. Set up GitHub account and create repositories
6. Contribute to open source projects
7. Learn Git workflows (Feature Branch, Gitflow)
8. Configure useful aliases and tools

Remember: Git has a learning curve, but it's worth the investment. Start with the basics and gradually explore advanced features as needed.
