"""Phase 0 characterization test — claude-sdk harness, one-shot prompt.

Runs ``omnigent run hello_world.yaml --harness claude-sdk -p
"..."`` as a real subprocess and snapshots the structural
observations (exit code, stderr absence, assistant text length).
Captured against current Omnigent; re-run unchanged in later
phases to prove the integration preserves behavior.

**What breaks if this fails:**
- Omnigent' ``ClaudeSDKExecutor`` regresses (auth, MCP tool
  bridging, Claude Code binary discovery, or the message-stream
  translation in ``claude_sdk_executor.run_turn``).
- ``_read_databrickscfg``'s PAT path regresses (the profile
  token becomes unreadable).
- ``omnigent.cli._run_agent`` for the ``-p`` one-shot path
  stops printing the assistant text to stdout on turn complete.
- The Claude Agent SDK dependency or the ``claude`` CLI binary
  goes missing from the Omnigent venv.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
per-harness suite.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Any

import pytest

from tests._model_pools import resolve_model
from tests.e2e.omnigent._snapshot import compare_snapshot

# Model + harness are hardcoded because the test name
# advertises "claude-sdk harness". A per-harness characterization
# test is meaningless without pinning the harness it covers.
_MODEL = resolve_model("databricks-claude-sonnet-4-6", key=__name__)
_HARNESS = "claude-sdk"
_PROMPT = "say hi in 5 words"

# Minimum assistant-text length — anything longer than "hi" is
# enough to prove the turn actually produced model output (not
# an empty response or a pure error banner).
_MIN_ASSISTANT_CHARS = 4

# Subprocess timeout. claude-sdk boots the Claude CLI plus an MCP
# bridge, so it's slower than openai-agents; 180s keeps headroom
# for cold starts on the CI host without letting a truly hung run
# pin the suite forever.
_RUN_TIMEOUT_SEC = 180


@pytest.fixture
def claude_sdk_available(omnigent_python: Path) -> bool:
    """
    Skip-guard for environments that can't run the claude-sdk
    harness.

    claude-sdk needs BOTH the Python package (inside the
    *omnigent* venv — the test's own venv is irrelevant because
    the test shells out) and the ``claude`` CLI binary on PATH.
    The binary is installed manually by users on their dev
    machines (``npm install -g @anthropic-ai/claude-code``), so
    CI environments commonly lack it.

    :param omnigent_python: The interpreter the subprocess
        uses. We probe THIS one for the ``claude_agent_sdk``
        import, not the current interpreter.
    :returns: True when both prerequisites are satisfied.
    """
    probe = subprocess.run(
        [
            str(omnigent_python),
            "-c",
            "import importlib.util, sys; "
            "sys.exit(0 if importlib.util.find_spec('claude_agent_sdk') else 1)",
        ],
        capture_output=True,
    )
    sdk_present = probe.returncode == 0
    cli_present = which("claude") is not None
    return sdk_present and cli_present


def test_per_harness_claude_sdk_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    patched_databrickscfg: None,
    claude_sdk_available: bool,
) -> None:
    """
    ``omnigent run hello_world.yaml --harness claude-sdk -p
    <prompt>`` exits 0 and emits a non-trivial assistant reply.

    Uses the ``patched_databrickscfg`` fixture to swap the
    active profile's section to a PAT for the duration of the
    test — necessary because ``ClaudeSDKExecutor`` reads the
    profile's ``token`` field directly and OAuth profiles
    produce 403s. This workaround is documented in the
    integration design doc and disappears once omnigent'
    ``_read_databrickscfg`` rewrite (audit item for phase 1)
    lands.

    :param omnigent_python: Interpreter with omnigent +
        claude-agent-sdk installed.
    :param omnigent_repo_root: Cwd for the subprocess.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` already populated from
        ``--llm-api-key``.
    :param patched_databrickscfg: Fixture that rewrites
        ``~/.databrickscfg`` to PAT form for the test and
        restores it on teardown.
    :param claude_sdk_available: True when the claude-sdk
        prerequisites (SDK package + ``claude`` binary) are
        present. If False, the test fails with an explicit
        reason rather than silently skipping — per the phase 0
        design, skip reasons must be explicit and environment
        gaps must be visible.
    """
    if not claude_sdk_available:
        pytest.fail(
            "claude-sdk harness prerequisites missing: both the "
            "'claude_agent_sdk' Python package and the 'claude' CLI "
            "binary must be present on PATH."
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

    # Claude SDK on macOS prints a one-line sandbox-fallback
    # warning to stderr on every launch (Linux-only bwrap).
    # That's benign and orthogonal to this test; the observation
    # we care about is "no hard errors in stderr" which we check
    # by excluding the known-benign line before the assertion.
    stderr_stripped = "\n".join(
        line
        for line in result.stderr.splitlines()
        if "Could not apply default local CLI sandbox" not in line
    ).strip()
    # Assistant text lands on stdout. It may be prefixed by an
    # echo of the prompt ("You> say hi...") depending on CLI
    # mode; the length check below tolerates either form.
    observed: dict[str, Any] = {
        "exit_code": result.returncode,
        "stderr_is_clean": stderr_stripped == "",
        # Trimmed because whitespace around LLM output is noisy
        # and not something we want the snapshot comparator to
        # trip on.
        "assistant_text": result.stdout.strip(),
    }

    # Full stderr surfaced on failure so CI logs show WHY the
    # run went wrong (e.g. 403 auth, missing binary) — stderr
    # here is opaque unless we dump it.
    diffs = compare_snapshot("test_per_harness_claude_sdk", observed)
    assert diffs == [], (
        "Snapshot mismatch for claude-sdk run:\n"
        + "\n".join(diffs)
        + f"\n\nstdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )
    # Separate assertion so the failure diagnostic names the
    # length-check directly instead of being buried in the
    # snapshot diff list.
    assert len(observed["assistant_text"]) >= _MIN_ASSISTANT_CHARS, (
        f"Claude SDK assistant text shorter than "
        f"{_MIN_ASSISTANT_CHARS} chars; got "
        f"{observed['assistant_text']!r}"
    )
