# Deployment Guide

This document describes how to deploy the technical knowledge base to GitHub Pages.

---

## Overview

This knowledge base is built with [mdBook](https://rust-lang.github.io/mdBook/), a Rust-based static site generator, and automatically deployed to GitHub Pages using GitHub Actions.

**Live Site**: [notes.paritystack.in](https://notes.paritystack.in)

---

## Technology Stack

- **Static Site Generator**: mdBook (Rust)
- **Hosting Platform**: GitHub Pages
- **Custom Domain**: notes.paritystack.in
- **CI/CD**: GitHub Actions
- **Build Output**: `./book` directory
- **Deployment Branch**: `gh-pages`

---

## Prerequisites

### For Local Development

1. **Rust & Cargo** (for mdBook installation)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **mdBook**
   ```bash
   cargo install mdbook
   ```

3. **Git**
   ```bash
   # Verify git is installed
   git --version
   ```

### For Deployment

- GitHub repository with GitHub Pages enabled
- Push access to the main branch
- GitHub Actions enabled (default for most repos)

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/paritystack/notes.git
cd notes
```

### 2. Install mdBook

```bash
cargo install mdbook
```

### 3. Serve Locally

```bash
mdbook serve
```

This will:
- Build the book from markdown files in `src/`
- Start a local server at http://localhost:3000
- Watch for file changes and auto-reload
- Display build output and any errors

### 4. Build Only

```bash
mdbook build
```

This generates static HTML files in the `./book` directory without starting a server.

### 5. Clean Build Output

```bash
mdbook clean
```

Removes the `./book` directory.

---

## Automated Deployment (GitHub Actions)

The repository uses GitHub Actions for automated deployment. Every push to the `main` branch triggers a build and deployment.

### Workflow Configuration

**File**: `.github/workflows/deploy.yml`

**Triggers**:
- Push to `main` branch → Build + Deploy
- Pull requests → Build only (no deployment)

**Process**:
1. Checkout repository
2. Setup mdBook (latest version)
3. Build static site (`mdbook build`)
4. Deploy to `gh-pages` branch (if on main)

### Manual Trigger

You can manually trigger the workflow from GitHub:
1. Go to **Actions** tab
2. Select **GitHub Pages** workflow
3. Click **Run workflow**

### View Deployment Status

1. Go to **Actions** tab in GitHub
2. Click on the latest workflow run
3. Check build logs for errors

---

## Manual Deployment

If you need to deploy manually (not recommended):

### Using GitHub CLI

```bash
# Build the book
mdbook build

# Install GitHub CLI if needed
# Ubuntu/Debian:
sudo apt install gh

# Authenticate
gh auth login

# Deploy using gh-pages branch
git checkout gh-pages
cp -r book/* .
git add .
git commit -m "Manual deployment"
git push origin gh-pages
```

### Using gh-pages npm package

```bash
# Build the book
mdbook build

# Install gh-pages
npm install -g gh-pages

# Deploy
gh-pages -d book
```

---

## GitHub Pages Configuration

### Enable GitHub Pages

1. Go to **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: **gh-pages** / **root**
4. Save

### Custom Domain Setup

The custom domain `notes.paritystack.in` is configured via:

1. **book.toml Configuration**
   ```toml
   [output.html]
   cname = "notes.paritystack.in"
   ```
   This creates a `CNAME` file in the build output.

2. **DNS Configuration**
   Add these DNS records at your domain registrar:
   ```
   Type: CNAME
   Name: notes
   Value: paritystack.github.io
   ```

3. **GitHub Pages Settings**
   - Go to **Settings** → **Pages**
   - Custom domain: `notes.paritystack.in`
   - Enforce HTTPS: ✅ Enabled

### SSL/TLS Certificate

GitHub Pages automatically provisions SSL certificates for custom domains. Allow 24 hours for certificate provisioning after DNS configuration.

---

## Deployment Checklist

Before deploying changes:

- [ ] Test locally with `mdbook serve`
- [ ] Verify all links work
- [ ] Check for markdown syntax errors
- [ ] Ensure code blocks render correctly
- [ ] Verify math equations display properly (MathJax)
- [ ] Review `src/SUMMARY.md` for correct navigation
- [ ] Check `book.toml` configuration
- [ ] Commit with descriptive message
- [ ] Push to main branch
- [ ] Monitor GitHub Actions workflow
- [ ] Verify deployment at notes.paritystack.in

---

## Configuration Files

### book.toml

Main configuration file for mdBook:

```toml
[book]
title = "My Notes"

[output.html]
cname = "notes.paritystack.in"  # Custom domain
mathjax-support = true          # Enable math rendering
```

### src/SUMMARY.md

Table of contents that defines the book structure. Any changes here affect navigation.

---

## Troubleshooting

### Build Failures

**Error: mdBook not found**
```bash
# Install or reinstall mdBook
cargo install mdbook --force
```

**Error: Broken links**
```bash
# mdBook will show which files have broken links
# Check the build output and fix markdown files
```

**Error: Math equations not rendering**
```bash
# Verify mathjax-support is enabled in book.toml
[output.html]
mathjax-support = true
```

### Deployment Issues

**Changes not appearing on live site**
1. Check GitHub Actions workflow status
2. Verify push was to `main` branch
3. Wait 1-2 minutes for deployment
4. Hard refresh browser (Ctrl+Shift+R)
5. Check if `gh-pages` branch was updated

**Custom domain not working**
1. Verify DNS records are correct
2. Check CNAME file exists in deployment
3. Wait for DNS propagation (up to 24 hours)
4. Verify custom domain in GitHub Pages settings

**404 errors**
1. Ensure `gh-pages` branch exists
2. Verify GitHub Pages source is set to `gh-pages` branch
3. Check that `book` directory is being deployed

**SSL certificate errors**
1. Wait 24 hours after DNS configuration
2. Remove and re-add custom domain in GitHub Pages settings
3. Check that HTTPS enforcement is enabled

### GitHub Actions Issues

**Workflow not triggering**
1. Verify `.github/workflows/deploy.yml` exists
2. Check GitHub Actions is enabled (Settings → Actions)
3. Ensure you pushed to `main` branch

**Permission denied errors**
1. Check workflow has `contents: write` permission
2. Verify `GITHUB_TOKEN` has sufficient permissions

**Build timeout**
1. Check for infinite loops in build process
2. Verify mdBook version is compatible
3. Review workflow logs for specific errors

---

## Performance Optimization

### Build Speed

- mdBook builds are typically fast (seconds)
- Large codebases may take longer
- Consider using mdBook plugins sparingly

### Site Performance

- mdBook generates static HTML (very fast)
- GitHub Pages uses CDN for global distribution
- Enable caching headers (automatic with GitHub Pages)
- Optimize images before adding to repository

---

## Monitoring

### Check Deployment Status

```bash
# View recent deployments
gh api repos/:owner/:repo/pages/builds

# View latest deployment
gh api repos/:owner/:repo/pages/builds/latest
```

### View Live Site

```bash
# Open in browser
open https://notes.paritystack.in

# Or use curl to check
curl -I https://notes.paritystack.in
```

---

## Rollback Procedure

If a deployment breaks the site:

### Method 1: Revert Git Commit

```bash
# Find the commit to revert to
git log --oneline

# Revert to previous commit
git revert <commit-hash>

# Push to trigger redeployment
git push origin main
```

### Method 2: Restore from gh-pages Branch

```bash
# Checkout gh-pages branch
git checkout gh-pages

# Find previous working commit
git log --oneline

# Reset to that commit
git reset --hard <commit-hash>

# Force push (use with caution)
git push --force origin gh-pages
```

---

## CI/CD Pipeline Details

### Workflow Stages

1. **Checkout** (actions/checkout@v3)
   - Clones repository with full history

2. **Setup mdBook** (peaceiris/actions-mdbook@v1)
   - Installs latest mdBook version
   - Caches installation for faster builds

3. **Build** (`mdbook build`)
   - Builds from `src/` directory
   - Outputs to `book/` directory
   - Fails on broken links by default

4. **Deploy** (peaceiris/actions-gh-pages@v4)
   - Only runs on main branch
   - Pushes `book/` directory to `gh-pages` branch
   - Uses `GITHUB_TOKEN` for authentication

### Deployment Time

- Build time: 10-30 seconds
- Deployment time: 5-15 seconds
- GitHub Pages propagation: 1-2 minutes
- Total: ~2-3 minutes from push to live

---

## Security Considerations

### GitHub Token Permissions

The workflow uses `GITHUB_TOKEN` with:
- `contents: write` - Required to push to gh-pages branch
- Scoped to repository only
- Automatically provided by GitHub Actions

### HTTPS Enforcement

- Always enabled for custom domains
- Redirects HTTP to HTTPS automatically
- Uses TLS 1.2+ for secure connections

### Branch Protection

Consider enabling branch protection for `main`:
1. Settings → Branches → Add rule
2. Require pull request reviews
3. Require status checks (GitHub Actions)
4. Prevent force pushes

---

## Best Practices

1. **Always test locally** before pushing
2. **Use descriptive commit messages** for easier rollbacks
3. **Monitor Actions workflow** after each push
4. **Keep mdBook updated** for security and features
5. **Use semantic versioning** for major changes
6. **Document breaking changes** in commit messages
7. **Review build logs** for warnings
8. **Test on multiple browsers** after deployment

---

## Additional Resources

- **mdBook Documentation**: https://rust-lang.github.io/mdBook/
- **GitHub Pages Docs**: https://docs.github.com/en/pages
- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **Custom Domain Setup**: https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site

---

## Support

For issues or questions:
1. Check this deployment guide
2. Review mdBook documentation
3. Check GitHub Actions workflow logs
4. Open an issue in the repository

---

**Last Updated**: November 2024
