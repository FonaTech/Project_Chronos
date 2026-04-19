# Contributing to Project Chronos

Thank you for your interest in contributing.

## Development Setup

```bash
git clone https://github.com/your-org/project-chronos
cd project-chronos
pip install -e ".[dev]"
```

## Running Tests

```bash
python tests/test_smoke.py
# or with pytest:
pytest tests/ -v
```

All 8 smoke tests must pass before submitting a PR.

## Code Style

```bash
ruff check chronos/ train_chronos.py
black chronos/ train_chronos.py
```

## Branch Strategy

- `main` — stable releases only
- `dev` — active development, PRs target here
- Feature branches: `feat/<name>`, bug fixes: `fix/<name>`

## Areas for Contribution

| Area | Description |
|------|-------------|
| `chronos/model/` | New attention variants, better LookaheadRouter architectures |
| `chronos/io/` | CUDA-native DMA prefetch, better clustering algorithms |
| `chronos/eval/` | MMLU/GSM8K integration, long-context benchmarks |
| `chronos/tuning/` | Better search spaces, multi-objective optimization |

## Reporting Issues

Please open a GitHub Issue with:
1. OS, GPU, Python/PyTorch versions
2. Minimal reproduction script
3. Full traceback

## Pull Request Checklist

- [ ] Tests pass (`python tests/test_smoke.py`)
- [ ] No ruff/black violations
- [ ] Docstring updated if public API changed
- [ ] `CHANGELOG.md` entry added for user-visible changes
