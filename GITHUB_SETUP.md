# GitHub Setup & Deployment Guide

## 🎉 Repository Ready for GitHub!

Your base-attentive repository is now fully set up and ready to be pushed to GitHub.

**Current Status:**
- ✅ Git repository initialized locally
- ✅ Initial commit created
- ✅ `develop` branch created and active
- ✅ All code, examples, and CI/CD workflows included
- ✅ Ready for remote push

---

## 📋 Step 1: Create GitHub Repository

### Option A: Using GitHub Web Interface

1. Go to https://github.com/new
2. Fill in:
   - **Repository name**: `base-attentive`
   - **Owner**: Select `earthai-tech` organization
   - **Description**: "BaseAttentive: Advanced Attention-based Forecasting Models"
   - **Visibility**: Public
   - **Do NOT initialize** with README, .gitignore, or license
3. Click "Create repository"

### Option B: Using GitHub CLI

```bash
gh repo create base-attentive --org earthai-tech --public --remote=origin --source=. --remote-name=origin --description="BaseAttentive: Advanced Attention-based Forecasting Models"
```

---

## 📤 Step 2: Add Remote and Push Code

### 2a. Add Remote Origin

```bash
cd d:\projects\base-attentive
git remote add origin https://github.com/earthai-tech/base-attentive.git
git branch -M main   # Rename master to main (optional but recommended)
```

### 2b. Push Master Branch

```bash
git push -u origin main   # or 'master' if you didn't rename
```

### 2c. Push Develop Branch

```bash
git push -u origin develop
```

### 2d. Verify

```bash
git branch -r
# Should show:
#   origin/main
#   origin/develop
```

---

## ⚙️ Step 3: Configure GitHub Actions Secrets

GitHub Actions needs credentials to publish to PyPI.

### For Test PyPI (Testing pre-release):

1. Go to https://test.pypi.org/account/tokens/
2. Create a new API token
3. In GitHub repo, go to **Settings → Secrets and variables → Actions**
4. Click **New repository secret**
5. Add:
   - **Name**: `TEST_PYPI_API_TOKEN`
   - **Value**: `pypi-...` (your token)

### For Production PyPI (Final release):

1. Go to https://pypi.org/account/tokens/
2. Create a new API token
3. Add to GitHub secrets:
   - **Name**: `PYPI_API_TOKEN`
   - **Value**: `pypi-...` (your token)

---

## 🔧 Step 4: Set Default Branch (Recommended)

1. Go to GitHub repo → **Settings → Branches**
2. Set **Default branch** to: `develop` (or keep as `main`)
3. Add branch protection rules:
   - **Branch name pattern**: `main`
   - **Require pull request reviews**: ✓
   - **Require status checks to pass**: ✓

---

## 🚀 Step 5: Enable Features

### GitHub Pages (for documentation later):

1. Go to **Settings → Pages**
2. **Source**: Deploy from branch
3. **Branch**: `gh-pages` (auto-created by Sphinx workflow)
4. **Folder**: `/ (root)`

### Code Quality (optional):

1. **Settings → Code security and analysis**
   - ✓ Enable Dependabot alerts
   - ✓ Enable Dependabot security updates
   - ✓ Enable secret scanning

---

## 📝 Step 6: Create Initial Issue & Discussions

### Create Welcome Issue:

```markdown
## 🎉 Welcome to BaseAttentive!

This is the initial tracking issue for the base-attentive repository.

### Phase 1: Core Setup (✅ COMPLETE)
- [x] Repository structure created
- [x] Example notebooks added
- [x] CI/CD pipelines configured
- [x] Documentation templates ready

### Phase 2: Component Migration (⏳ IN PROGRESS)
- [ ] Copy attention components from geoprior-v3
- [ ] Copy encoder/decoder components
- [ ] Copy utilities and validators
- [ ] Implement full BaseAttentive

### Phase 3: Documentation (⏳ PLANNED)
- [ ] Build Sphinx documentation
- [ ] Add API reference
- [ ] Deploy to ReadTheDocs

### Phase 4: Release (⏳ PLANNED)
- [ ] Publish to TestPyPI
- [ ] Community feedback
- [ ] Publish to PyPI

Please contribute! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
```

---

## 📚 Complete Push Sequence

Run this in your terminal:

```bash
# Navigate to repository
cd d:\projects\base-attentive

# Add remote
git remote add origin https://github.com/earthai-tech/base-attentive.git

# Push master/main
git push -u origin master
# OR if you renamed to main:
# git branch -M main
# git push -u origin main

# Push develop
git push -u origin develop

# Verify
git branch -r
git remote -v
```

---

## ✅ Post-Push Checklist

After pushing to GitHub, verify:

- [ ] Both `master` and `develop` branches visible on GitHub
- [ ] All workflows appear in `.github/workflows/`
- [ ] All documentation files display correctly
- [ ] Example notebooks are viewable
- [ ] **Actions** tab shows workflow triggers
- [ ] Code security settings are configured

---

## 🔄 Continuous Integration Status

### On Each Push:

1. **Tests Workflow** (`tests.yml`)
   - Runs pytest on Python 3.9-3.12
   - Tests on Windows, macOS, Linux
   - Builds distribution package
   - Uploads coverage to Codecov

2. **Code Quality** (`code-quality.yml`)
   - Black formatting check
   - isort import sorting
   - Bandit security scan
   - Radon complexity analysis

3. **Documentation** (`documentation.yml`)
   - Builds Sphinx docs (when added)
   - Uploads as artifact

4. **Release to PyPI** (`pypi-release.yml`)
   - Triggered on GitHub Release creation
   - Automatically publishes to PyPI

---

## 📖 Quick Start Commands

```bash
# Clone from GitHub
git clone https://github.com/earthai-tech/base-attentive.git
cd base-attentive

# Checkout develop
git checkout develop

# Install in development mode
pip install -e ".[dev]"

# Run tests locally
pytest tests/ -v

# Format code
black src/ tests/

# Check imports
isort src/ tests/

# Create a feature branch
git checkout -b feature/your-feature-name
git push -u origin feature/your-feature-name

# Create pull request on develop branch
# (via GitHub web interface)
```

---

## 🎯 Next Steps After GitHub Push

1. **Check Actions**: Go to repo → Actions tab
   - Verify workflows run successfully
   - Fix any CI/CD issues

2. **Set Default Branch**:
   - Settings → Branches → Set `develop` or `main` as default

3. **Add Branch Protection**:
   - Require pull request reviews
   - Require status checks to pass

4. **Begin Phase 2**:
   - Copy components from geoprior-v3
   - Create feature branches for each component
   - Submit pull requests for code review

5. **Invite Contributors**:
   - Add team members to the organization
   - Set up code review process

---

## 🐛 Troubleshooting

### "fatal: remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/earthai-tech/base-attentive.git
```

### "Permission denied (publickey)"
- Ensure SSH key is added: `ssh -T git@github.com`
- Or use HTTPS with personal access token instead

### CI/CD Workflows Not Triggering
- Ensure `.github/workflows/*.yml` files are pushed
- Check branch protection isn't blocking them
- Try pushing to develop branch

### Coverage Reports Not Uploading
- Add Codecov token to GitHub secrets (optional)
- Or leave public (recommended for open source)

---

## 📊 Repository Statistics

After push, you'll see:
- **Issues**: 0 (create initial tracking issue)
- **Pull Requests**: 0 (ready for contributions)
- **Releases**: 0 (ready for first release)
- **Workflows**: 4 CI/CD pipelines
- **Branches**: 2+ (main/master + develop)

---

## 🎁 What You Have

```
✅ Production-ready repository structure
✅ Comprehensive CI/CD pipelines (tests, code quality, docs, PyPI)
✅ Example notebooks (3 tutorials)
✅ Contributing guidelines
✅ Code of conduct
✅ Apache 2.0 license
✅ Ready for team collaboration
✅ Automated testing on 3 OS, 4 Python versions
✅ Coverage reporting
✅ Automated PyPI release on tag
```

---

**Ready to push? Follow the steps above and you're all set! 🚀**
