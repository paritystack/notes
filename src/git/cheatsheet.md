# Git Cheatsheet

## Git Commands

### Basic setup

```bash
git config --global user.name "<name>"
git config --global user.email "<email>"
git config --global core.editor "vi"
```

### Create a new branch from an orphan branch

```bash
git switch --orphan <new branch>
git commit --allow-empty -m "Initial commit on orphan branch"
git push -u origin <new branch>
```
