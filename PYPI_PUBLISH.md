# Publishing to PyPI

This guide explains how to publish the Hepatitis C Predictor package to PyPI.

## Prerequisites

1. Create accounts on:
   - PyPI: https://pypi.org/account/register/
   - TestPyPI (for testing): https://test.pypi.org/account/register/

2. Install build tools:
   ```bash
   pip install build twine
   ```

## Build the Package

1. Clean previous builds:
   ```bash
   rm -rf dist/ build/ *.egg-info
   ```

2. Build the package:
   ```bash
   python -m build
   ```

This creates distribution files in the `dist/` directory.

## Test on TestPyPI (Recommended)

1. Upload to TestPyPI:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

2. Test installation:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ hepatitis-c-predictor
   ```

3. Test the CLI:
   ```bash
   hepatitis-c-demo
   ```

## Publish to PyPI

Once tested, publish to the real PyPI:

```bash
python -m twine upload dist/*
```

## Installation from PyPI

After publishing, users can install with:

```bash
pip install hepatitis-c-predictor
```

And launch the demo with:

```bash
hepatitis-c-demo
```

Or use it programmatically:

```python
from src.models import HepatitisNet, load_model
from src.data import load_raw_data, clean_data

# Load data
data = load_raw_data()
cleaned_data, encoder = clean_data(data)

# Load pretrained model
model, info = load_model('models/hepatitis_model.pth')
```

## Version Bumping

To release a new version:

1. Update version in `pyproject.toml`:
   ```toml
   version = "0.1.1"
   ```

2. Rebuild and upload:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

## API Token (Recommended)

For security, use API tokens instead of passwords:

1. Generate token on PyPI: https://pypi.org/manage/account/token/
2. Create `~/.pypirc`:
   ```ini
   [pypi]
   username = __token__
   password = pypi-AgEIcHlwaS5vcmc...
   ```

## Troubleshooting

- **Package name already exists**: Choose a different name in `pyproject.toml`
- **Missing dependencies**: Ensure all dependencies are listed in `pyproject.toml`
- **Import errors**: Check `[tool.setuptools.packages.find]` configuration
