# Common Git Scenarios & Questions

A practical Q&A guide for developers working with Git. Organized by task and includes solutions, alternatives, and common pitfalls.

## Undoing Changes

### How do I discard changes in my working directory?

**Use case:** You've made edits to files but want to throw them away completely and start over.

**Solution:**
```bash
# Discard changes in a single file
git restore file.txt

# Discard all changes in working directory
git restore .

# Alternative (older syntax)
git checkout -- file.txt
git checkout -- .
```

**Gotchas:**
- This is **permanent** — changes are lost and cannot be recovered
- Only affects your working directory, not staged or committed changes
- Use `git diff` first to see what you're about to lose

**Learn more:** See the "Undoing Changes" section in [README.md](index.html#undoing-changes)

---

### How do I unstage files I've already added?

**Use case:** You ran `git add` but now realize you don't want some files in the next commit.

**Solution:**
```bash
# Unstage a specific file
git restore --staged file.txt

# Unstage all files
git restore --staged .

# Alternative (older syntax)
git reset HEAD file.txt
git reset HEAD
```

**Gotchas:**
- The file remains modified in your working directory — only the staging area is affected
- If you want to discard the changes entirely, run `git restore file.txt` after unstaging

**Learn more:** See the "Staging Area" section in [README.md](index.html#staging-area)

---

### How do I undo my last commit but keep my changes?

**Use case:** You committed something but want to make more changes before committing again, or you want to split it into multiple commits.

**Solution:**
```bash
# Undo last commit, keep changes in working directory
git reset --soft HEAD~1

# Now make more changes or stage selectively
git add file1.txt
git commit -m "First part"
git add file2.txt
git commit -m "Second part"
```

**Alternatives:**
- `git reset --mixed HEAD~1` (same as `--soft` but also unstages files)
- `git rebase -i HEAD~1` (if you want to edit the commit message instead)

**Gotchas:**
- `--soft` keeps changes **staged** (in the index)
- `--mixed` (default) keeps changes but **unstages** them (moves to working directory)
- `--hard` **discards changes permanently** — be careful!
- This only works safely on unpushed commits

**Learn more:** See the "Commits" section in [README.md](index.html#commits)

---

### How do I undo a commit that's already been pushed?

**Use case:** You pushed a commit to shared branch but realized it has a bug or shouldn't be there.

**Solution:**
```bash
# Create a new commit that undoes the changes (safe for shared branches)
git revert commit-hash

# Then push the revert
git push origin branch-name
```

**Alternatives:**
- If the branch is yours and nobody else is using it:
  ```bash
  git reset --hard HEAD~1
  git push --force-with-lease origin branch-name
  ```

**Gotchas:**
- **Never use `--force` on main/master or shared branches** — use `--force-with-lease` if you must
- `git revert` creates a new commit that undoes changes — this is safer for shared code
- `git reset --hard` + force push rewrites history — only do this on feature branches
- If multiple people are on the branch, force push will break their local repos

**Learn more:** See the "Commits" section in [README.md](index.html#commits)

---

### How do I revert only one file in an existing commit?

**Use case:** A commit changed multiple files, but you only want to undo changes to one of them — bringing that single file back to how it looked *before* the commit while leaving the other files untouched.

**Solution:**
```bash
# Amend the last commit (NOT yet pushed):
# Pull the file as it was in the parent commit (this also stages it)
git checkout HEAD~1 -- path/to/file.txt
git commit --amend --no-edit

# For any commit, including one already pushed (creates a NEW commit — safe for shared branches):
git checkout <commit-hash>~1 -- path/to/file.txt
git commit -m "Revert path/to/file.txt to its state before <commit-hash>"
```

**Alternatives:**
```bash
# Modern equivalent of the checkout step (Git 2.23+)
git restore --source=<commit-hash>~1 -- path/to/file.txt
git commit -m "Revert path/to/file.txt"
```

**Gotchas:**
- `git checkout <commit>~1 -- file` pulls the file from the *parent* of that commit and stages it — it does not touch any other file
- The `--amend` approach rewrites the last commit (changes its hash) — only safe if not yet pushed
- The new-commit approach is safe for shared/pushed branches
- This brings the file back to its pre-commit state; it does not selectively undo only part of the file's diff (use `git checkout -p` for that)

---

### How do I remove a file that was already committed?

**Use case:** You accidentally committed a secret file (.env, keys, etc.) or large file that shouldn't be in Git.

**Solution:**
```bash
# Remove file from Git but keep it locally
git rm --cached file.txt

# Add it to .gitignore
echo "file.txt" >> .gitignore

# Commit the removal
git commit -m "Remove file.txt from tracking"
```

**For sensitive data already pushed:**
```bash
# Recommended: git filter-repo (modern, fast — install separately)
git filter-repo --invert-paths --path secrets.txt

# Or use BFG Repo-Cleaner (also fast and easy)
bfg --delete-files secrets.txt

# Legacy: git filter-branch (built-in but slow and discouraged)
git filter-branch --tree-filter 'rm -f secrets.txt' HEAD
```

**Gotchas:**
- `git rm --cached` removes from Git but keeps the file locally — use this for `.gitignore` fixes
- `git filter-branch` rewrites all history — affects everyone on the repo; Git now discourages it in favor of `git filter-repo`
- For sensitive data, consider it **compromised** even after removal (check git log)
- Push with `--force-with-lease` after rewriting history

**Learn more:** See the "Troubleshooting" section in [README.md](index.html#troubleshooting)

---

### How do I fix a wrong commit message?

**Use case:** You committed with a typo or unclear message, or want to add more detail.

**Solution:**
```bash
# Fix the most recent commit message
git commit --amend -m "Corrected message"

# Or open in editor for longer messages
git commit --amend

# If already pushed, force push the correction
git push --force-with-lease origin branch-name
```

**Alternatives:**
- Use `git rebase -i HEAD~3` to edit messages from past commits

**Gotchas:**
- Amending rewrites the commit — changes the hash
- Only amend commits that haven't been pushed or are on your feature branch
- If pushed to shared branch, discuss with team before force-pushing
- Use `--no-edit` to amend without changing the message: `git commit --amend --no-edit`

---

## Working with History

### How do I view the commit history?

**Use case:** You need to understand what changes were made and when.

**Solution:**
```bash
# Basic log
git log

# Compact, one-line per commit
git log --oneline

# With graph visualization
git log --graph --oneline --all

# Show last N commits
git log -n 10

# Show commits by specific author
git log --author="John"

# Show commits in date range
git log --since="2 weeks ago" --until="2024-01-01"

# Search commit messages
git log --grep="fix"

# Show commits affecting specific file
git log -- file.txt
```

**Gotchas:**
- `git log` shows only current branch by default — add `--all` to see all branches
- `--oneline` is great for quick scanning but hides details
- Use `git log -p` to see actual code changes (can be verbose)

**Learn more:** See the "Viewing History" section in [README.md](index.html#viewing-history)

---

### How do I see what changed in a specific commit?

**Use case:** You want to understand what a particular commit did.

**Solution:**
```bash
# Show changes in specific commit
git show commit-hash

# Show just the files changed
git show --name-only commit-hash

# Show stats (how many lines changed)
git show --stat commit-hash

# Compare specific file across commits
git diff commit1 commit2 -- file.txt
```

**Gotchas:**
- `git show` displays full diff — can be long for large commits
- Use `--stat` for a summary of changes
- Commit hash can be abbreviated (first 7 characters usually work)

---

### How do I find which commit introduced a bug?

**Use case:** You know a bug exists, but don't know when it was introduced.

**Solution:**
```bash
# Start bisect session
git bisect start

# Mark current version as bad
git bisect bad

# Mark a known good commit
git bisect good v1.0

# Git will check out middle commit — test and mark as good/bad
git bisect bad  # or git bisect good

# Continue until Git narrows it down
# Once found, end the session
git bisect reset
```

**Alternatives:**
- Use `git blame file.txt` to see who changed each line and when
- Use `git log --oneline file.txt` to see all commits affecting a file

**Gotchas:**
- Bisect requires good test case — you need to confirm if a commit is good or bad
- It's a binary search, so much faster than manually checking commits
- Don't forget `git bisect reset` or you'll stay in bisect mode

**Learn more:** See the "Bisect" section in [README.md](index.html#bisect)

---

### How do I squash multiple commits into one?

**Use case:** You have several small "WIP" or "fix" commits that should be combined into a single commit before merging.

**Solution:**
```bash
# Interactive rebase for last 3 commits
git rebase -i HEAD~3

# In the editor:
# pick first-commit
# squash second-commit
# squash third-commit

# Save and edit the combined commit message
git push --force-with-lease origin branch-name
```

**Alternatives:**
```bash
# Soft reset to combine all changes, then commit once
git reset --soft HEAD~3
git commit -m "Combined feature"
```

**Gotchas:**
- Interactive rebase rewrites history — only safe on unpushed commits
- Use `--force-with-lease` instead of `--force` to avoid overwriting others' work
- Can cause merge conflicts if others have based work on your commits
- Practice on a test branch first if unfamiliar

---

### How do I reorder or remove commits from history?

**Use case:** You want to reorganize commits or remove a "oops" commit before merging.

**Solution:**
```bash
# Interactive rebase for last 5 commits
git rebase -i HEAD~5

# In editor, you can:
# - Reorder lines to reorder commits
# - Change "pick" to "drop" to remove a commit
# - Change "pick" to "reword" to edit message
# - Change "pick" to "squash" to combine with previous

# After editing, save and resolve any conflicts
git rebase --continue

# Push with force-with-lease
git push --force-with-lease origin branch-name
```

**Gotchas:**
- Rewriting history is permanent once pushed — discuss with team first
- If conflicts occur during rebase, resolve them and `git rebase --continue`
- Use `git rebase --abort` if you mess up
- Never rebase on main/master or shared branches

---

### How do I see all changes I made to a file?

**Use case:** You want to understand all modifications to a specific file across its history.

**Solution:**
```bash
# Show all commits affecting a file
git log -- file.txt

# Show all changes (diffs) to a file
git log -p -- file.txt

# Show who changed each line (blame)
git blame file.txt

# Show changes in a date range
git log --since="1 month ago" -p -- file.txt

# Show changes by specific author
git log --author="John" -p -- file.txt
```

**Gotchas:**
- `-p` can produce very long output for old files — consider limiting with `-n 10`
- `git blame` shows the most recent change per line, not all history
- Large files with many changes can be slow to process

---

## Branches & Merging

### How do I create a new branch and switch to it?

**Use case:** You want to start working on a feature in isolation without affecting the main branch.

**Solution:**
```bash
# Create and switch to new branch (modern syntax, Git 2.23+)
git switch -c feature/my-feature

# Alternative (older syntax)
git checkout -b feature/my-feature

# Create branch without switching
git branch feature/my-feature
git switch feature/my-feature
```

**Gotchas:**
- New branch starts from your current branch's latest commit
- Branch name convention: `feature/`, `bugfix/`, `hotfix/` prefixes are common
- Create from `main` or `develop` depending on your workflow
- Don't forget to switch to the branch before making changes

---

### How do I switch between branches?

**Use case:** You need to work on a different branch or check out main to pull updates.

**Solution:**
```bash
# Switch to existing branch (modern syntax)
git switch branch-name

# Alternative (older syntax)
git checkout branch-name

# Switch to previous branch
git switch -

# Create and switch in one command
git switch -c new-branch
```

**Gotchas:**
- You can switch with uncommitted changes — Git carries them over — *unless* they'd be overwritten by files that differ between the branches
- If the switch is blocked ("local changes would be overwritten"), commit or stash them first
- `git switch -` toggles between current and previous branch (useful for switching back)

**Learn more:** See the "Branches" section in [README.md](index.html#branches)

---

### How do I delete a branch?

**Use case:** Feature is merged and you want to clean up the branch.

**Solution:**
```bash
# Delete local branch
git branch -d feature/my-feature

# Force delete if not fully merged
git branch -D feature/my-feature

# Delete remote branch
git push origin --delete feature/my-feature
git push origin :feature/my-feature  # Alternative syntax

# Clean up remote tracking branches
git fetch --prune origin
```

**Gotchas:**
- `-d` only works if branch is fully merged — use `-D` to force delete
- Deleting locally doesn't delete on remote — you must push the deletion
- After deleting remote, other developers' local copies still have the branch
- Use `git fetch --prune` to remove stale remote tracking branches

---

### How do I merge a feature branch into main?

**Use case:** Feature is complete and tested, ready to integrate into main.

**Solution:**
```bash
# Make sure main is up to date
git checkout main
git pull origin main

# Switch to feature branch and rebase it
git checkout feature/my-feature
git rebase main

# Resolve any conflicts if they occur
git rebase --continue

# Switch back to main and merge
git checkout main
git merge feature/my-feature

# Push to remote
git push origin main

# Delete feature branch
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```

**Alternatives:**
```bash
# Fast-forward merge (simpler history)
git merge feature/my-feature

# Non-fast-forward merge (creates merge commit)
git merge --no-ff feature/my-feature
```

**Gotchas:**
- Always pull `main` first to avoid pushing old commits
- Rebasing the feature branch before merge keeps history clean
- Merge conflicts must be resolved before merge completes
- Use `--no-ff` if you want a merge commit for visibility
- Fast-forward merge is simpler but loses feature branch history

**Learn more:** See the "Merging" section in [README.md](index.html#merging)

---

### How do I resolve a merge conflict?

**Use case:** Git can't automatically merge changes from two branches because they modified the same lines.

**Solution:**
```bash
# Start merge (will show conflicts)
git merge feature/branch

# Check which files have conflicts
git status

# Open conflicted file — look for markers:
# <<<<<<< HEAD
# Your changes
# =======
# Their changes
# >>>>>>> feature/branch

# Edit the file to resolve conflict
# Remove conflict markers and keep the code you want

# Stage the resolved file
git add resolved-file.txt

# Complete the merge
git commit -m "Resolve merge conflict"

# Or use merge tool
git mergetool
```

**Alternatives:**
```bash
# Abort merge if you don't want to continue
git merge --abort

# Resolve by taking all of one side
git checkout --ours file.txt    # Keep your changes
git checkout --theirs file.txt  # Keep their changes
```

**Gotchas:**
- Merge markers (`<<<<<<<`, `=======`, `>>>>>>>`) must be removed manually
- Removing entire sections can cause logic errors — understand the code
- After resolving all files, run tests before committing
- `git mergetool` can help if many conflicts, but manual editing is often clearer

**Learn more:** See the "Handling Merge Conflicts" section in [README.md](index.html#handling-merge-conflicts)

---

### How do I rebase my feature branch on main?

**Use case:** Main branch has new commits, and you want your feature branch to start from the latest main instead of your old base.

**Solution:**
```bash
# Fetch the latest remote state (updates origin/main)
git fetch origin

# Switch to feature branch
git switch feature/my-feature

# Rebase onto the up-to-date remote main
git rebase origin/main

# Resolve conflicts if they occur, then continue
git rebase --continue

# Force push to remote (since rebase rewrites history)
git push --force-with-lease origin feature/my-feature
```

**Alternatives:**
```bash
# Use merge instead of rebase (creates merge commit)
git merge origin/main

# Interactive rebase (combine with cleanup)
git rebase -i origin/main
```

**Gotchas:**
- Rebasing rewrites history — only do on your own feature branches
- Use `--force-with-lease` not `--force` to avoid overwriting others
- If someone else is on the branch, rebase will break their local copy
- Conflicts may occur during rebase — resolve with `git rebase --continue`
- Use `git rebase --abort` if something goes wrong

**Learn more:** See the "Rebasing" section in [README.md](index.html#rebasing)

---

### How do I rename a branch?

**Use case:** You named a branch incorrectly or want to follow a naming convention.

**Solution:**
```bash
# Rename current branch
git branch -m new-name

# Rename specific branch (without switching)
git branch -m old-name new-name

# Update remote (delete old, push new)
git push origin --delete old-name
git push origin -u new-name
```

**Gotchas:**
- Renaming doesn't affect merged code — just the branch name
- Other developers will still have the old branch name locally
- Must update remote explicitly — local rename alone isn't enough
- Use `-u` flag when pushing to set upstream tracking

---

## Remote Operations

### How do I push my local commits to the remote?

**Use case:** You've made commits locally and want to share them with the team.

**Solution:**
```bash
# Push current branch to remote
git push origin branch-name

# Set upstream and push (first time on new branch)
git push -u origin branch-name

# Push all branches
git push --all origin

# Push specific tag
git push origin tag-name

# Push all tags
git push --tags
```

**Gotchas:**
- First push of new branch requires `-u` flag to set upstream
- Can't push if you're behind remote — pull first with `git pull --rebase`
- `--force` and `--force-with-lease` should be avoided on shared branches
- Check that you're pushing to the correct remote (`origin` is typical)

**Learn more:** See the "Pushing" section in [README.md](index.html#pushing)

---

### How do I pull the latest changes from remote?

**Use case:** Team members have pushed changes and you want to update your local repository.

**Solution:**
```bash
# Pull current branch (fetch + merge)
git pull origin branch-name

# Pull with rebase instead of merge
git pull --rebase origin branch-name

# Fetch without merging (safer to review first)
git fetch origin

# Update all remote tracking branches
git fetch --all
```

**Alternatives:**
```bash
# Two-step approach (more control)
git fetch origin
git merge origin/branch-name
```

**Gotchas:**
- Pull does fetch + merge in one step — can create unexpected merge commits
- Use `--rebase` to avoid merge commits and keep linear history
- Always pull before pushing to avoid conflicts
- Fetching alone doesn't merge — use it to review before pulling

**Learn more:** See the "Fetching and Pulling" section in [README.md](index.html#fetching-and-pulling)

---

### How do I keep my fork in sync with the original repository?

**Use case:** You forked a repo and original has new changes you want to incorporate.

**Solution:**
```bash
# Add upstream remote (if not already done)
git remote add upstream https://github.com/original-owner/repo.git

# Fetch from upstream
git fetch upstream

# Switch to main and merge
git switch main
git merge upstream/main

# Push to your fork
git push origin main

# Update feature branches
git switch feature/my-feature
git rebase main
git push --force-with-lease origin feature/my-feature
```

**Gotchas:**
- Must add upstream remote first (one-time setup)
- Merging upstream/main into your main is safe
- Feature branches may need rebase if they conflict with upstream changes
- Keep your fork updated before creating pull requests

**Learn more:** See the "Syncing Fork with Upstream" section in [README.md](index.html#syncing-fork-with-upstream)

---

### How do I collaborate with someone on the same branch?

**Use case:** Multiple people are working on the same feature branch and need to share work.

**Solution:**
```bash
# Person A: Create and push branch
git switch -c feature/shared-feature
git add .
git commit -m "Initial implementation"
git push -u origin feature/shared-feature

# Person B: Clone the branch
git fetch origin
git switch feature/shared-feature

# Person B: Make changes and push
git add .
git commit -m "Add tests"
git push origin feature/shared-feature

# Person A: Pull updates
git pull origin feature/shared-feature

# Continue collaborating...
```

**Gotchas:**
- Both people might edit same files — merge conflicts likely
- Pull frequently to avoid large conflicts
- Communicate via Slack/comments about who's working on what
- Use meaningful commit messages so others understand changes
- Consider pair programming for complex sections

**Learn more:** See the "Collaborating on a Branch" section in [README.md](index.html#collaborating-on-a-branch)

---

### How do I create a pull request on GitHub?

**Use case:** Feature is ready for review, you want to request merge into main.

**Solution:**
```bash
# 1. Make sure branch is up to date and pushed
git checkout main
git pull origin main
git checkout feature/my-feature
git rebase main
git push --force-with-lease origin feature/my-feature

# 2. On GitHub:
# - Navigate to repository
# - Click "Compare & pull request" or "New pull request"
# - Select base branch (main) and compare branch (feature/my-feature)
# - Add title and description
# - Click "Create pull request"

# 3. Respond to review comments
git add .
git commit -m "Address review feedback"
git push origin feature/my-feature

# 4. After approval and merge, cleanup
git switch main
git pull origin main
git branch -d feature/my-feature
git push origin --delete feature/my-feature
```

**Gotchas:**
- Rebase before PR to ensure clean history
- Write clear PR description explaining why changes are needed
- Be responsive to review comments
- Don't force push after review starts (unless requested)
- Test locally before creating PR

---

### How do I handle a force push that rewrote my branch?

**Use case:** Someone force-pushed changes to a shared branch and your local copy is out of sync.

**Solution:**
```bash
# Fetch the new version
git fetch origin

# See what changed
git log origin/branch-name

# Reset to match remote
git reset --hard origin/branch-name

# Alternative: Create new branch from remote
git switch -c branch-name-local origin/branch-name
```

**Gotchas:**
- Force push from others indicates history rewrite — check with them first
- `reset --hard` **discards local changes** — make sure nothing is uncommitted
- Communicate about force pushes — can disrupt team workflow
- Don't force push to main or shared branches without agreement

---

## Advanced Techniques

### How do I cherry-pick a specific commit from another branch?

**Use case:** A bug fix was made in one branch but you need it in another branch, or you want to apply one commit without merging entire branch.

**Solution:**
```bash
# Apply single commit to current branch
git cherry-pick commit-hash

# Cherry-pick multiple commits
git cherry-pick commit1 commit2

# Cherry-pick a range (note: commit1..commit3 EXCLUDES commit1)
git cherry-pick commit1..commit3

# Cherry-pick a range INCLUDING the first commit
git cherry-pick commit1^..commit3

# Cherry-pick without committing (for review)
git cherry-pick -n commit-hash

# Abort if conflicts or mistakes
git cherry-pick --abort
```

**Alternatives:**
```bash
# If conflicts occur during cherry-pick
git cherry-pick --continue  # After resolving conflicts
```

**Gotchas:**
- Cherry-pick creates a new commit (with new hash) — not the same as the original
- Can cause duplicate commits if the same commit exists in both branches
- Use sparingly — if you need many commits, consider merge instead
- Conflicts may occur if the commit depends on other commits

**Learn more:** See the "Cherry-Pick" section in [README.md](index.html#cherry-pick)

---

### How do I stash my changes temporarily?

**Use case:** You're in the middle of work but need to switch branches or pull updates, without committing yet.

**Solution:**
```bash
# Stash current changes
git stash

# Stash with descriptive message (modern syntax; `git stash save` is deprecated)
git stash push -m "WIP: working on auth feature"

# List all stashes
git stash list

# Apply most recent stash
git stash apply

# Apply specific stash
git stash apply stash@{2}

# Apply and remove stash
git stash pop

# Delete stash
git stash drop stash@{0}

# Stash including untracked files
git stash -u

# Create branch from stash
git stash branch feature/from-stash
```

**Gotchas:**
- Stash is temporary and local — not pushed to remote
- `apply` keeps stash; `pop` removes it after applying
- Untracked files are not stashed by default — use `-u` flag
- Stashes can accumulate — clean them up periodically
- Merge conflicts can occur when applying stash if code has changed

**Learn more:** See the "Stashing" section in [README.md](index.html#stashing)

---

### How do I recover a deleted branch?

**Use case:** You accidentally deleted a branch but want to restore it.

**Solution:**
```bash
# View recent commits including deleted branches
git reflog

# Find the commit hash of the branch you deleted
# Create new branch from that commit
git switch -c recovered-branch commit-hash

# Or push it back to remote
git push origin recovered-branch
```

**Alternatives:**
```bash
# If you know the commit message
git log --all --grep="keyword" --oneline

# Search reflog for branch name
git reflog | grep "branch-name"
```

**Gotchas:**
- Reflog only keeps history for a limited time (default 90 days)
- Must act quickly if branch was deleted long ago
- Reflog is local only — won't help if remote branch is deleted
- If remote branch exists, easier to re-clone from remote

**Learn more:** See the "Recovering from Mistakes" section in [README.md](index.html#recovering-from-mistakes)

---

### How do I undo a force push or hard reset?

**Use case:** You force-pushed or did a hard reset and realized you lost commits.

**Solution:**
```bash
# Check recent commits including discarded ones
git reflog

# Find the commit you want to recover
git switch -c recovered-branch commit-hash

# Or reset back to it
git reset --hard commit-hash
git push --force-with-lease origin branch-name
```

**Alternatives:**
```bash
# If the commit still exists on remote
git fetch origin
git reset --hard origin/branch-name
```

**Gotchas:**
- Reflog is local only — if another machine was affected, they can't recover
- Must act within 90 days (default reflog expiration)
- This is why `--force-with-lease` is safer than `--force`
- Communicate with team immediately if force push affects shared branch

**Learn more:** See the "Recovering from Mistakes" section in [README.md](index.html#recovering-from-mistakes)

---

### How do I see unpushed commits?

**Use case:** You want to know what commits you have locally that haven't been pushed yet.

**Solution:**
```bash
# Show commits ahead of remote
git log origin/branch-name..HEAD

# Or more readable
git log origin/branch-name..HEAD --oneline

# Show commits behind remote
git log HEAD..origin/branch-name

# See both directions
git log --oneline --decorate --all

# Check status relative to remote
git status
```

**Gotchas:**
- Local and remote can drift apart if multiple people are pushing
- Use this before `git push` to review what you're about to push
- Fetch first to ensure remote tracking branches are up to date

---

### How do I use git blame to find who changed a line?

**Use case:** You want to know who last modified a specific line and why.

**Solution:**
```bash
# Show who changed each line
git blame file.txt

# Show specific line range
git blame -L 10,20 file.txt

# Ignore whitespace-only changes when assigning blame
git blame -w file.txt

# Show the author's email instead of name
git blame -e file.txt

# Find the commit that changed a line
git blame file.txt | grep "line content"

# Then view that commit
git show commit-hash
```

**Gotchas:**
- Shows most recent change per line, not all history
- If line was reformatted, blame shows the formatter not the original author
- Use `git log -p -- file.txt` to see all changes to a file
- Blame can be slow on large files with long history

---

### How do I split a commit into multiple commits?

**Use case:** You committed too many unrelated changes together and want to split them.

**Solution:**
```bash
# Start interactive rebase
git rebase -i HEAD~1

# Change "pick" to "edit" for the commit to split
# Save and exit

# When Git stops at that commit:
git reset HEAD~1  # Undo the commit but keep changes

# Stage and commit the first part
git add file1.txt file2.txt
git commit -m "Part 1: Feature implementation"

# Stage and commit the second part
git add file3.txt
git commit -m "Part 2: Tests"

# Continue rebase
git rebase --continue

# Force push
git push --force-with-lease origin branch-name
```

**Gotchas:**
- Interactive rebase rewrites history — only safe on unpushed commits
- Requires careful staging — use `git add -p` for partial file staging
- Conflicts can occur if other commits depend on the split commit
- Test thoroughly after splitting

---

### How do I create and push a tag?

**Use case:** You want to mark a release or important version in your code.

**Solution:**
```bash
# Create lightweight tag
git tag v1.0.0

# Create annotated tag (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0"

# Tag a past commit
git tag v1.0.0 commit-hash

# List tags
git tag

# Push single tag
git push origin v1.0.0

# Push all tags
git push --tags

# Delete local tag
git tag -d v1.0.0

# Delete remote tag
git push origin --delete v1.0.0

# Checkout a tag
git switch --detach v1.0.0
```

**Gotchas:**
- Lightweight tags are just references; annotated tags are full objects
- Always push tags explicitly — they don't push by default
- Deleting tags locally doesn't delete from remote
- Tags typically don't change — avoid retagging same version

**Learn more:** See the "Tags" section in [README.md](index.html#tags)

---

### How do I find when a line was deleted?

**Use case:** A line of code existed in the past but is gone now, and you want to find when it was removed.

**Solution:**
```bash
# Search commit history for deleted lines
git log -S "search text" -- file.txt

# Show which commit removed it
git log -S "search text" -p -- file.txt

# More readable output
git log -S "search text" --oneline -- file.txt

# Then view that commit
git show commit-hash
```

**Gotchas:**
- `-S` searches for addition or removal of text (pickaxe)
- Use quotes if searching for code with special characters
- Can be slow on large repos with long history
- Search is case-sensitive by default

