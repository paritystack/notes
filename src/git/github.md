# GitHub

GitHub is a web-based platform that provides hosting for Git repositories along with collaboration features, CI/CD, project management, and more.

## Quick Start

### SSH Setup

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Start SSH agent
eval "$(ssh-agent -s)"

# Add SSH key to agent
ssh-add ~/.ssh/id_ed25519

# Copy public key to clipboard
cat ~/.ssh/id_ed25519.pub
# Then paste in GitHub Settings → SSH and GPG keys → New SSH key
```

### Clone Repository

```bash
# Using SSH (recommended)
git clone git@github.com:<username>/<repository>.git

# Using HTTPS
git clone https://github.com/<username>/<repository>.git

# Create new branch
git switch -c <new_branch>

# Push branch to remote
git push -u origin <new_branch>
```

## Pull Request Workflow

### Creating a Pull Request

```bash
# 1. Create and switch to feature branch
git checkout -b feature/new-feature

# 2. Make changes and commit
git add .
git commit -m "feat: Add new feature"

# 3. Push branch to GitHub
git push -u origin feature/new-feature

# 4. Open browser and create PR
# Navigate to repository → Pull requests → New pull request
# Select your branch → Create pull request

# Or use GitHub CLI
gh pr create --title "Add new feature" --body "Description of changes"
```

### Working on Pull Request

```bash
# After creating PR, make additional commits
git add .
git commit -m "Address feedback"
git push origin feature/new-feature

# Update PR with latest main
git checkout main
git pull origin main
git checkout feature/new-feature
git rebase main
git push --force-with-lease origin feature/new-feature

# Request review
gh pr review <pr-number> --request-changes --body "Please fix..."
gh pr review <pr-number> --approve --body "LGTM!"
```

### Reviewing Pull Requests

```bash
# Check out PR locally
gh pr checkout <pr-number>
# Or manually:
git fetch origin pull/<pr-number>/head:pr-<pr-number>
git checkout pr-<pr-number>

# Test changes
npm test
npm run build

# Add review comments
gh pr review <pr-number> --comment --body "Looks good!"

# Approve PR
gh pr review <pr-number> --approve

# Request changes
gh pr review <pr-number> --request-changes --body "Please address..."
```

### Merging Pull Requests

```bash
# Merge via GitHub CLI
gh pr merge <pr-number> --merge
gh pr merge <pr-number> --squash
gh pr merge <pr-number> --rebase

# Via web interface:
# - Merge commit: Preserves all commits
# - Squash and merge: Combines all commits into one
# - Rebase and merge: Adds commits to base branch

# After merging, cleanup
git checkout main
git pull origin main
git branch -d feature/new-feature
```

## GitHub Issues

### Creating Issues

```bash
# Create issue via CLI
gh issue create --title "Bug: Login fails" --body "Description of bug"

# Create with labels
gh issue create --title "Feature request" --label "enhancement"

# List issues
gh issue list
gh issue list --label "bug"
gh issue list --assignee "@me"

# View issue
gh issue view <issue-number>
```

### Working with Issues

```bash
# Assign issue
gh issue edit <issue-number> --add-assignee "@me"

# Add labels
gh issue edit <issue-number> --add-label "bug,high-priority"

# Close issue
gh issue close <issue-number>

# Reopen issue
gh issue reopen <issue-number>

# Link PR to issue (in commit or PR description)
git commit -m "Fix login bug

Fixes #123"
# Or "Closes #123", "Resolves #123"
```

### Issue Templates

Create `.github/ISSUE_TEMPLATE/bug_report.md`:

```markdown
---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
A clear description of what you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g. Ubuntu 22.04]
- Browser: [e.g. Chrome 120]
- Version: [e.g. v1.2.3]
```

## GitHub Actions (CI/CD)

### Basic Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Install dependencies
      run: npm ci

    - name: Run tests
      run: npm test

    - name: Run linter
      run: npm run lint

    - name: Build
      run: npm run build
```

### Action Permissions

```bash
# Enable workflow permissions
# Repository Settings → Actions → General → Workflow permissions
# Select: Read and write permissions

# Or in workflow file
permissions:
  contents: write
  pull-requests: write
  issues: write
```

### Deployment Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [ main ]
  release:
    types: [ published ]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Build
      run: npm run build

    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./build
        cname: yourdomain.com
```

### Useful GitHub Actions

```yaml
# Test on multiple Node versions
strategy:
  matrix:
    node-version: [16, 18, 20]

# Cache dependencies
- name: Cache node modules
  uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}

# Create release
- name: Create Release
  uses: actions/create-release@v1
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  with:
    tag_name: ${{ github.ref }}
    release_name: Release ${{ github.ref }}

# Comment on PR
- name: Comment on PR
  uses: actions/github-script@v6
  with:
    script: |
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: 'Build successful! ✅'
      })
```

## GitHub Pages

### Setup GitHub Pages

```bash
# 1. Create gh-pages branch
git checkout --orphan gh-pages
git rm -rf .
echo "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>" > index.html
git add index.html
git commit -m "Initial GitHub Pages commit"
git push origin gh-pages

# 2. Enable in repository settings
# Settings → Pages → Source → Select branch: gh-pages, folder: /(root)

# 3. Add custom domain (optional)
echo "yourdomain.com" > CNAME
git add CNAME
git commit -m "Add custom domain"
git push origin gh-pages
```

### Deploy with Actions

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Build
      run: |
        npm ci
        npm run build

    - name: Deploy
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./build
        cname: yourdomain.com  # Optional
```

## GitHub CLI (gh)

### Installation

```bash
# macOS
brew install gh

# Linux (Debian/Ubuntu)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh

# Authenticate
gh auth login
```

### Common gh Commands

```bash
# Repository
gh repo create <name>
gh repo clone <repo>
gh repo view
gh repo fork

# Pull Requests
gh pr create
gh pr list
gh pr view <number>
gh pr checkout <number>
gh pr merge <number>
gh pr diff <number>
gh pr review <number>

# Issues
gh issue create
gh issue list
gh issue view <number>
gh issue close <number>

# Releases
gh release create v1.0.0
gh release list
gh release download v1.0.0

# Workflows
gh workflow list
gh workflow run <workflow>
gh run list
gh run view <run-id>

# Gists
gh gist create <file>
gh gist list
```

## Collaboration Workflows

### Fork and Contribute

```bash
# 1. Fork repository on GitHub (click Fork button)

# 2. Clone your fork
gh repo fork <original-repo> --clone

# 3. Add upstream remote
git remote add upstream https://github.com/original-owner/repo.git

# 4. Create feature branch
git checkout -b feature/my-contribution

# 5. Make changes and commit
git add .
git commit -m "Add feature"

# 6. Keep fork updated
git fetch upstream
git checkout main
git merge upstream/main
git push origin main

# 7. Push feature branch
git push origin feature/my-contribution

# 8. Create pull request
gh pr create --base main --head feature/my-contribution
```

### Code Review Best Practices

```bash
# As PR author:
# - Keep PRs small and focused
# - Write clear description
# - Link related issues
# - Respond to feedback promptly

# As reviewer:
# - Review promptly
# - Be constructive and specific
# - Test changes locally
# - Approve or request changes

# Request specific reviewers
gh pr create --reviewer @username1,@username2

# Check PR status
gh pr status

# View PR checks
gh pr checks <pr-number>
```

### Team Collaboration

```bash
# Protect branches
# Repository Settings → Branches → Add branch protection rule
# - Require pull request reviews
# - Require status checks to pass
# - Require branches to be up to date
# - Include administrators

# Add collaborators
# Repository Settings → Collaborators → Add people

# Use code owners (.github/CODEOWNERS)
# Require approval from code owners
* @team-name
/docs/ @docs-team
*.js @frontend-team
```

## Project Management

### GitHub Projects

```bash
# Create project
gh project create --title "My Project"

# Add issues to project
gh issue create --project "My Project"

# View project
gh project view <project-number>
```

### Milestones

```bash
# Create milestone
# Issues → Milestones → New milestone

# Assign issue to milestone
gh issue edit <number> --milestone "v1.0"

# View milestone progress
# Issues → Milestones
```

## Security Features

### Dependabot

Enable in Settings → Security → Dependabot:
- Dependabot alerts
- Dependabot security updates
- Dependabot version updates

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
```

### Secret Scanning

```bash
# Enable in Settings → Security → Code security and analysis
# - Secret scanning
# - Secret scanning push protection

# Use secrets in workflows
steps:
  - name: Deploy
    env:
      API_KEY: ${{ secrets.API_KEY }}
    run: deploy.sh
```

### Security Advisories

```bash
# Create security advisory
# Security → Advisories → New draft security advisory

# Report vulnerability privately
# Contact repository maintainers through security tab
```

## Webhooks and API

### Setup Webhook

```bash
# Repository Settings → Webhooks → Add webhook
# Payload URL: https://your-server.com/webhook
# Content type: application/json
# Events: Push, Pull request, Issues, etc.
```

### GitHub API

```bash
# Using curl
curl -H "Authorization: token YOUR_TOKEN" \
  https://api.github.com/user/repos

# Using gh CLI with API
gh api repos/<owner>/<repo>/pulls

gh api graphql -f query='
  query {
    repository(owner: "owner", name: "repo") {
      pullRequests(first: 10) {
        nodes {
          title
          number
        }
      }
    }
  }
'
```

## Advanced Features

### GitHub Codespaces

```bash
# Create codespace
gh codespace create --repo <repo>

# List codespaces
gh codespace list

# Connect to codespace
gh codespace ssh
```

### GitHub Packages

Publish package:

```yaml
- name: Publish to GitHub Packages
  run: npm publish
  env:
    NODE_AUTH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### GitHub Discussions

```bash
# Enable in Settings → General → Features → Discussions

# Create discussion
gh api repos/<owner>/<repo>/discussions \
  -f title="Discussion title" \
  -f body="Discussion body"
```

## Best Practices

1. **Branch Protection**: Enable branch protection on main/develop
2. **Required Reviews**: Require at least one approval before merging
3. **Status Checks**: Require CI/CD to pass before merging
4. **Linear History**: Use squash or rebase merging for clean history
5. **Signed Commits**: Enable commit signing for security
6. **Templates**: Use PR and issue templates for consistency
7. **Labels**: Use labels to categorize issues and PRs
8. **Milestones**: Track progress with milestones
9. **Projects**: Use GitHub Projects for project management
10. **Documentation**: Keep README and CONTRIBUTING.md updated