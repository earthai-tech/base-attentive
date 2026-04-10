# BaseAttentive Contribution Guidelines

Thank you for your interest in contributing to BaseAttentive!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/base-attentive.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Install in development mode: `pip install -e ".[dev]"`

## Development Workflow

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
```bash
# Format with black
black src/ tests/

# Check with flake8
flake8 src/ tests/

# Type check with mypy
mypy src/
```

### Creating A Pull Request

1. Push your changes to your fork
2. Submit a pull request to the `develop` branch
3. Ensure all CI/CD checks pass
4. Address any review comments

## Reporting Issues

Please use GitHub Issues to report bugs. Include:
- Python version and environment details
- Steps to reproduce
- Error messages and tracebacks
- Expected vs actual behavior

## Code of Conduct

We follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).

Thank you for contributing! 🎉
