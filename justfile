# List available recipes
default:
    just --list

# Install/sync the dev environment
install:
    uv sync --extra dev

# Update uv.lock
lock:
    uv lock

# Run all code-quality checks (mirrors .github/workflows/python-package.yml)
check:
    uv run --extra dev ruff format --check .
    uv run --extra dev ruff check .
    uv run --extra dev mypy .
    uv run --extra dev pytest tests/ --verbose

# Auto-fix formatting and lint issues
fmt:
    uv run --extra dev ruff format .
    uv run --extra dev ruff check --fix .

# Run the test suite only
test:
    uv run --extra dev pytest tests/ --verbose

# Bump the version, tag, and push — triggers .github/workflows/publish.yml
# which runs the tests, builds, publishes to PyPI, and creates the GitHub release.
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    just check
    sed -i 's/^version = ".*"/version = "{{version}}"/' pyproject.toml
    sed -i 's/^__version__ = ".*"/__version__ = "{{version}}"/' __init__.py
    just lock
    git add pyproject.toml __init__.py uv.lock
    git commit -m "bump version to {{version}}"
    git tag "v{{version}}"
    git push origin main
    git push origin "v{{version}}"

# Verify a release landed: GitHub release + PyPI package version (defaults to pyproject.toml's version)
verify version=`grep -m1 '^version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/'`:
    gh release view "v{{version}}"
    echo "---pypi---"
    curl -s https://pypi.org/pypi/climate-change/json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['info']['version'])"
