# BaseAttentive Badges Guide

This document explains all the badges used in the BaseAttentive project and their current status.

## Badge Status Overview

### Continuous Integration (CI/CD) Badges

#### Tests Badge
![Tests](https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml/badge.svg?branch=develop)

- **Status**: Tracks test suite execution
- **Branch**: develop
- **When it updates**: After each push to develop branch
- **What it tracks**: 
  - Unit tests (pytest)
  - Multi-platform testing (Ubuntu, Windows, macOS)
  - Multiple Python versions (3.10, 3.11, 3.12)
  - Code coverage reports

**How to trigger**: Push changes to the develop branch

---

#### Code Quality Badge
![Code Quality](https://github.com/earthai-tech/base-attentive/actions/workflows/code-quality.yml/badge.svg?branch=develop)

- **Status**: Tracks code quality checks
- **Branch**: develop
- **When it updates**: After each push to develop branch
- **What it tracks**:
  - Ruff linting (formatting and style)
  - Security checks with bandit
  - Code complexity analysis with radon
  - TODO/FIXME detection

**How to trigger**: Push changes to the develop branch

---

#### Documentation Badge
![Documentation](https://readthedocs.org/projects/base-attentive/badge/?version=latest)

- **Status**: Tracks documentation build status on ReadTheDocs
- **When it updates**: After documentation push or webhook trigger
- **What it tracks**:
  - Sphinx documentation generation
  - ReStructuredText compilation
  - Link validation
  - API documentation generation

**How to trigger**: 
1. Push changes to develop branch
2. ReadTheDocs automatically builds the docs
3. Visit https://base-attentive.readthedocs.io to view

**Note**: The ReadTheDocs badge will show "passing" or "failing" after the first successful build.

---

#### PyPI Release Badge
![Publish](https://github.com/earthai-tech/base-attentive/actions/workflows/pypi-release.yml/badge.svg?branch=master)

- **Status**: Tracks PyPI release workflow
- **Branch**: master (stable releases only)
- **When it updates**: On version tags or release events
- **What it tracks**:
  - Package build process
  - Distribution file generation
  - PyPI upload status

**How to trigger**: 
1. Create a GitHub Release
2. Tag it with version number (e.g., v1.0.0)
3. Workflow automatically builds and can publish to PyPI

---

### Release & Version Badges

#### Version Badge
![Version](https://img.shields.io/github/v/release/earthai-tech/base-attentive?display_name=tag&sort=semver)

- **Status**: Shows latest GitHub release version
- **When it updates**: When a new GitHub Release is created
- **What it shows**: Latest release tag (e.g., v1.0.0)

**How to trigger**: Create a GitHub Release with a version tag

---

#### License Badge
![License](https://img.shields.io/github/license/earthai-tech/base-attentive)

- **Status**: Static badge showing license
- **What it shows**: Apache-2.0 license
- **Updates**: Only when LICENSE file changes

---

### Code Quality & Metrics Badges

#### Coverage Badge
![Coverage](https://codecov.io/gh/earthai-tech/base-attentive/branch/develop/graph/badge.svg)

- **Status**: Test coverage percentage
- **Branch**: develop
- **When it updates**: After successful test runs with coverage reports
- **What it shows**: Percentage of code covered by tests

**How to trigger**: 
1. Ensure tests pass with coverage reporting
2. Coverage data is sent to Codecov.io
3. Visit https://codecov.io/gh/earthai-tech/base-attentive to see details

**Note**: Requires Codecov configuration and repository setup

---

#### Code Style Badge
![Code Style](https://img.shields.io/badge/code%20style-ruff-46a758.svg)

- **Status**: Static badge indicating code style tool
- **What it shows**: Uses Ruff for code formatting
- **Configuration**: .github/workflows/code-quality.yml enforces ruff

---

#### Python Version Badge
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)

- **Status**: Minimum supported Python version
- **What it shows**: Python 3.10 or higher required
- **Tested versions**: 3.10, 3.11, 3.12

---

## Badge Status Meaning

| Badge Status | Meaning |
|---|---|
| ✓ Passing (green) | All checks passed successfully |
| ✗ Failed (red) | One or more checks failed |
| ? No status (gray) | Workflow has not run yet |
| Building (yellow) | Workflow is currently running |

## How Badges Get Their Status

### GitHub Actions Badges
1. When code is pushed to a tracked branch, GitHub Actions automatically runs the workflow
2. The workflow executes all defined steps (tests, linting, etc.)
3. If all steps succeed → badge shows ✓ passing
4. If any step fails → badge shows ✗ failed
5. The badge URL persists and updates with each workflow run

### ReadTheDocs Badge
1. ReadTheDocs monitors the repository via webhook
2. When code changes, it automatically builds the documentation
3. If build succeeds → badge shows ✓ passing
4. If build fails → badge shows ✗ failed
5. Takes 1-5 minutes after push to see status change

### Codecov Badge
1. Test jobs upload coverage.xml file to Codecov.io
2. Codecov calculates coverage percentage and stores history
3. Badge displays current coverage percentage
4. Updates after each successful test run

## Troubleshooting Badge Status

### Badge shows "no status"
**Possible causes:**
- Workflow has never run successfully (first time setup)
- Workflow is failing due to errors
- GitHub Actions is still running the workflow

**Solution:**
1. Go to repository's "Actions" tab
2. Find the workflow that's not showing status
3. Check the latest run for errors
4. Fix the errors and push again
5. Wait 1-2 minutes for badge to update

### Workflow keeps failing
**Common issues and solutions:**

1. **Tests failing**
   - Run `pytest tests/ -v` locally to debug
   - Check Python version compatibility
   - Ensure all dependencies are installed

2. **Linting failing**
   - Run `ruff check src/` to see issues
   - Run `ruff format src/` to auto-fix
   - Commit fixed code

3. **Documentation failing**
   - Run `cd docs && make html` locally
   - Check for Sphinx errors
   - Validate ReStructuredText syntax

4. **Dependencies missing**
   - Update `pyproject.toml [project.optional-dependencies]`
   - Ensure workflow installs correct extras: `pip install -e ".[dev]"`

## Setting Up Codecov Integration

To enable the Coverage badge:

1. Go to https://codecov.io
2. Sign in with GitHub
3. Add the `earthai-tech/base-attentive` repository
4. Codecov will automatically detect coverage reports from GitHub Actions
5. Coverage badge will start displaying after first successful test run

## Setting Up ReadTheDocs

To enable the Documentation badge:

1. Go to https://readthedocs.org
2. Sign in with GitHub
3. Import the `base-attentive` project
4. ReadTheDocs will automatically build docs from `docs/` folder
5. Badge will show status after first build (takes 1-5 minutes)

## Badge Status Check Commands

You can locally verify badge requirements before pushing:

```bash
# Run tests
pytest tests/ -v 

# Run linting
ruff check src/ tests/
ruff format src/ tests/

# Build documentation
cd docs && make html

# Check coverage
pytest tests/ --cov=src/base_attentive --cov-report=html

# Check build distribution
python -m build
twine check dist/*
```

## Quick Reference URLs

| Badge | Status Check |
|-------|--------------|
| Tests | https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml |
| Code Quality | https://github.com/earthai-tech/base-attentive/actions/workflows/code-quality.yml |
| Documentation | https://base-attentive.readthedocs.io |
| Coverage | https://codecov.io/gh/earthai-tech/base-attentive |
| Releases | https://github.com/earthai-tech/base-attentive/releases |

## Notes

- Badges update automatically after each GitHub Actions workflow completes
- Direct status checks typically take 30 seconds - 2 minutes
- ReadTheDocs can take 2-5 minutes to process documentation changes
- First workflow run may take longer due to dependency installation
- Coverage badge requires Codecov integration and successful test run with coverage reports
