# Contributing to Omnigent

Thanks for your interest in improving Omnigent. Issues and pull requests are
welcome. For larger changes, open an issue first so we can discuss the approach.

Please don't include secrets, internal URLs, customer data, or private
configuration in issues, tests, examples, or logs.

## Development setup

This is a Python package with an optional frontend under `ap-web/`. Use
[`uv`](https://docs.astral.sh/uv/) for local development:

```bash
git clone https://github.com/omnigent-ai/omnigent.git
cd omnigent

uv python install
uv venv --python "$(cat .python-version)"
uv sync --extra all --extra dev
source .venv/bin/activate    # or prefix commands with `uv run`
```

Common checks:

```bash
uv run pytest                      # Python tests (e2e/live skipped by default)
uv run ruff check . && uv run ruff format --check .
uv run pre-commit run --all-files
```

When touching `ap-web/`:

```bash
cd ap-web && npm install && npm run lint && npm run build
```

## Pull requests

- Branch from `main`, keep changes focused, and include tests or docs when relevant.
- Sign off your commits with `git commit -s` (Developer Certificate of Origin).
