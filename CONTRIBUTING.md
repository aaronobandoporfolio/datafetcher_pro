# Contributing to DataFetcherPro

Thank you for considering contributing! This document provides guidelines for contributing to DataFetcherPro.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, Python version, package versions)
- **Code samples** if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List some examples** of how it would be used

### Pull Requests

1. **Fork the repo** and create your branch from `develop`
2. **Add tests** if you've added code that should be tested
3. **Update documentation** if you've changed APIs
4. **Ensure tests pass**: `pytest tests/ -v`
5. **Format your code**: `black .`
6. **Lint your code**: `flake8 datafetcher_pro/ --max-line-length=120`
7. **Write clear commit messages**

#### Pull Request Process

1. Update README.md with details of changes if needed
2. Update CHANGELOG.md with your changes under [Unreleased]
3. The PR will be merged once you have sign-off from maintainers

## Development Setup
```bash
# Clone your fork
git clone https://github.com/aaronobandoporfolio/datafetcher_pro.git
cd datafetcher-pro
```

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .

# Run tests
pytest tests/ -v --cov=datafetcher_pro

## Support the Project

If you find DataFetcherPro useful and would like to support its development, you can contribute financially:

- [Buy me a coffee](buymeacoffee.com/aaronobandk)
- [PayPal](https://paypal.me/AaronObando505?locale.x=en_US&country.x=CR)

Your support helps cover hosting, tools, and development time. Every contribution is appreciated! :D
