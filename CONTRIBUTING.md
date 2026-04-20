# Contributing

Thanks for your interest in ECG-Explain.

This is primarily a portfolio / research project, but issues and pull requests
are welcome — particularly:

- Bug fixes
- Improvements to interpretability methods
- Cross-dataset validation results
- Documentation improvements

## Development setup

    git clone https://github.com/M-Omarjee/ecg-explain.git
    cd ecg-explain
    uv sync --all-extras
    uv run pre-commit install   # auto-runs lint + format on commit

Run the test suite before opening a PR:

    make test
    make lint
    make typecheck

## Code style

- Ruff handles linting and formatting (configured in `pyproject.toml`).
- Type hints required for all public functions in `src/`.
- Tests required for new features.

## Reporting issues

Please include:

- Python version (`python --version`)
- OS
- Full error traceback
- Minimal reproduction steps
