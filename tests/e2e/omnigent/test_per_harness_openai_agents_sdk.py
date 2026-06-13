"""Phase 0 characterization test — openai-agents-sdk harness, one-shot prompt.

Runs ``omnigent run hello_world.yaml --harness openai-agents
--model <gpt-model> -p "..."`` as a real subprocess and snapshots
structural observations (exit code, stderr cleanliness, assistant
text length). Captured against current Omnigent; re-run
unchanged in later phases to prove the integration preserves
behavior for the openai-agents harness.

**What breaks if this fails:**
- Omnigent' ``OpenAIAgentsSDKExecutor`` regresses (the Runner
  lifecycle, the Responses-API adapter in
  ``omnigent.open_responses_sdk``, the MCP tool bridging, or
  the event stream translation to ``ExecutorEvent`` types).
- The ``openai-agents`` Python package (``agents`` module) is
  missing from the omnigent venv or its public API changes
  incompatibly.
- The Databricks model-serving gateway at
  ``OPENAI_BASE_URL`` rejects requests that previously worked
  (token invalid, model decommissioned, etc.).
- ``omnigent.cli._run_agent`` for the ``-p`` one-shot path
  stops printing the assistant text on turn complete.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
per-harness suite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests._model_pools import resolve_model
from tests.e2e.omnigent._snapshot import compare_snapshot

# Model + harness are hardcoded because the test name advertises
# "openai-agents harness"; a per-harness characterization test is
# meaningless without pinning the harness it covers.
# databricks-gpt-5-4-mini is the gateway model documented in the
# repo-level CLAUDE.md for OpenAI-flavored harnesses.
_MODEL = resolve_model("databricks-gpt-5-4-mini", key=__name__)
_HARNESS = "openai-agents"
_PROMPT = "say hi in 5 words"

# Minimum assistant-text length. Anything longer than "hi" proves
# the turn produced a real model reply rather than an empty
# response or a pure error banner.
_MIN_ASSISTANT_CHARS = 4

# Subprocess timeout. openai-agents runs inside the harness
# process (no external subprocess to boot) so it's faster than
# codex/claude-sdk, but 180s keeps headroom for cold starts on
# loaded CI hosts.
_RUN_TIMEOUT_SEC = 180


@pytest.fixture
def openai_agents_available(omnigent_python: Path) -> bool:
    """
    Availability probe for the openai-agents-sdk harness.

    ``OpenAIAgentsSDKExecutor`` imports the ``agents`` package
    lazily on first use. The package must be installed in the
    *omnigent* venv (the subprocess interpreter) — the current
    pytest interpreter is irrelevant because the test shells out.

    :param omnigent_python: Interpreter the subprocess will
        use. Probe THIS one for the ``agents`` import, not the
        current pytest interpreter.
    :returns: True when the ``agents`` package imports cleanly
        in the omnigent venv.
    """
    probe = subprocess.run(
        [
            str(omnigent_python),
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('agents') else 1)",
        ],
        capture_output=True,
    )
    return probe.returncode == 0


def test_per_harness_openai_agents_sdk_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    openai_agents_available: bool,
) -> None:
    """
    ``omnigent run hello_world.yaml --harness openai-agents -p
    <prompt>`` exits 0 and emits a non-trivial assistant reply.

    Does NOT use ``patched_databrickscfg`` because the
    openai-agents executor honors ``OPENAI_BASE_URL`` /
    ``OPENAI_API_KEY`` env vars directly (populated by
    ``omnigent_credentials_env``) — no ``~/.databrickscfg``
    touch is required for this harness. This matches the pattern
    the design doc calls out: openai-agents is the "cleanest" of
    the harnesses re: credential plumbing.

    :param omnigent_python: Interpreter with omnigent +
        ``openai-agents`` installed.
    :param omnigent_repo_root: Cwd for the subprocess.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` populated from
        ``--llm-api-key``.
    :param openai_agents_available: True when the ``agents``
        package is importable in the omnigent venv. On False
        the test fails with an explicit reason per CLAUDE.md
        rule 30 (no silent skips).
    """
    if not openai_agents_available:
        pytest.fail(
            "openai-agents-sdk harness prerequisite missing: "
            "the 'agents' Python package (openai-agents) must be "
            "installed in the Omnigent venv."
        )

    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "hello_world.yaml"

    result = subprocess.run(
        [
            str(omnigent_python),
            "-m",
            "omnigent",
            "run",
            str(yaml_path),
            "--model",
            _MODEL,
            "--harness",
            _HARNESS,
            "-p",
            _PROMPT,
            "--no-log",
            "--no-session",
        ],
        env=omnigent_credentials_env,
        cwd=str(omnigent_repo_root),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )

    observed: dict[str, Any] = {
        "exit_code": result.returncode,
        "stderr_is_clean": result.stderr.strip() == "",
        # Trimmed because whitespace around LLM output is noisy
        # and not something we want the snapshot comparator to
        # trip on.
        "assistant_text": result.stdout.strip(),
    }

    # Full stderr surfaced on failure so CI logs show WHY the run
    # went wrong — stderr here is opaque unless we dump it.
    diffs = compare_snapshot(
        "test_per_harness_openai_agents_sdk",
        observed,
    )
    assert diffs == [], (
        "Snapshot mismatch for openai-agents run:\n"
        + "\n".join(diffs)
        + f"\n\nstdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )
    # Separate assertion so a length regression names the length
    # check directly instead of being buried in the snapshot diff.
    assert len(observed["assistant_text"]) >= _MIN_ASSISTANT_CHARS, (
        f"openai-agents assistant text shorter than "
        f"{_MIN_ASSISTANT_CHARS} chars; got "
        f"{observed['assistant_text']!r}"
    )
