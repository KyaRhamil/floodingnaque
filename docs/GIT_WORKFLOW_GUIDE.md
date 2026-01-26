# Git Workflow Guide for Floodingnaque

This guide explains how to commit and push changes to the repository with GPG signing enabled.

## Quick Reference

```powershell
# Stage and commit (GPG signing is automatic)
git add .
git commit -m "type(scope): description"

# Push to current branch
git push
```

---

## Your Setup

| Setting | Value |
|---------|-------|
| GPG Key | `AABA93F5EEE0F76E` |
| Auto-sign commits | ✅ Enabled globally |
| Repository rule | Requires verified (signed) commits |

---

## Daily Workflow

### For Small Changes (bug fixes, minor updates)

You can commit directly to a feature branch:

```powershell
# 1. Make sure you're on the right branch
git checkout chore/ci-optimization  # or create new: git checkout -b fix/bug-name

# 2. Make your changes, then stage them
git add .

# 3. Commit (GPG signing happens automatically)
git commit -m "fix: resolve login timeout issue"

# 4. Push
git push
```

### For Big Changes (new features, refactoring)

**Yes, create a new branch** for significant changes:

```powershell
# 1. Start from main (or your base branch)
git checkout main
git pull origin main

# 2. Create a new branch
git checkout -b feat/new-feature-name

# 3. Make changes and commit as you go
git add .
git commit -m "feat: add user authentication"

# 4. Push the branch (first time needs upstream)
git push -u origin feat/new-feature-name

# 5. Create a Pull Request on GitHub
```

---

## Branch Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feat/description` | `feat/flood-prediction-api` |
| Bug fix | `fix/description` | `fix/database-connection` |
| Chore | `chore/description` | `chore/update-dependencies` |
| Documentation | `docs/description` | `docs/api-reference` |
| Refactor | `refactor/description` | `refactor/service-layer` |

---

## Commit Message Format

This repository uses [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes nor adds |
| `test` | Adding tests |
| `chore` | Maintenance tasks |
| `ci` | CI/CD changes |
| `perf` | Performance improvement |

### Examples

```powershell
git commit -m "feat(api): add flood prediction endpoint"
git commit -m "fix(auth): resolve token expiration issue"
git commit -m "docs: update API documentation"
git commit -m "chore: upgrade dependencies"
```

---

## When to Create a Branch?

### ✅ Create a new branch when:

- Adding a new feature
- Making breaking changes
- Refactoring large portions of code
- Working on something that takes multiple commits
- Collaborating with others on the same feature
- You want code review before merging

### ❌ You might skip branching when:

- Fixing a typo
- Updating documentation
- Single-line config changes
- **But note:** Your repo requires signed commits on all branches, so you can still commit to `main` if branch protection allows

---

## Troubleshooting

### "Commits must have verified signatures" error

Your commits aren't being signed. Check:

```powershell
# Verify GPG signing is enabled
git config --global commit.gpgsign
# Should output: true

# Verify your signing key
git config --global user.signingkey
# Should output: A982934110ED36B14F52FCADAABA93F5EEE0F76E

# Test GPG signing
echo "test" | gpg --clearsign
```

### GPG key expired or not working

```powershell
# List your keys
gpg --list-secret-keys --keyid-format=long

# If needed, generate a new key and add to GitHub
gpg --full-generate-key
```

### Push rejected after rebase

If you rebased and need to push:

```powershell
git push --force-with-lease
```

⚠️ **Only use force push on your own feature branches, never on `main`**

---

## Complete Example: New Feature Workflow

```powershell
# 1. Start fresh from main
git checkout main
git pull origin main

# 2. Create feature branch
git checkout -b feat/weather-alerts

# 3. Work and commit incrementally
# ... make changes ...
git add .
git commit -m "feat(alerts): add weather alert model"

# ... more changes ...
git add .
git commit -m "feat(alerts): implement alert notification service"

# 4. Push branch
git push -u origin feat/weather-alerts

# 5. Go to GitHub and create Pull Request
# 6. After review and approval, merge via GitHub UI
# 7. Clean up local branch
git checkout main
git pull origin main
git branch -d feat/weather-alerts
```

---

## Summary

| Action | Command |
|--------|---------|
| Create branch | `git checkout -b type/name` |
| Stage changes | `git add .` |
| Commit (auto-signed) | `git commit -m "type: message"` |
| Push new branch | `git push -u origin branch-name` |
| Push existing branch | `git push` |
| Switch branches | `git checkout branch-name` |
| Update from remote | `git pull origin main` |
