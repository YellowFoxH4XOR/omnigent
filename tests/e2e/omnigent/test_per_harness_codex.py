"""Phase 0 characterization test — codex harness, one-shot prompt.

Runs ``omnigent run hello_world.yaml --harness codex --model
<codex-compatible-model> -p "..."`` as a real subprocess and
snapshots structural observations (exit code, stderr cleanliness,
assistant text length). Captured against current Omnigent;
re-run unchanged in later phases to prove the integration
preserves behavior for the codex harness.

**What breaks if this fails:**
- Omnigent' ``CodexExecutor`` regresses (``codex app-server``
  subprocess orchestration, App Server JSON-RPC protocol, the
  message-stream translation in ``codex_executor.run_turn``, or
  the Databricks config-override generation in
  ``_databricks_codex_config_overrides``).
- The ``codex`` CLI binary disappears from PATH or its
  ``app-server`` subcommand changes its startup contract.
- ``omnigent.databricks_executor._read_databrickscfg`` regresses
  for the PAT path — ``CodexExecutor`` transitively depends on it
  to resolve the Databricks host + token for the model proxy.
- ``omnigent.cli._run_agent`` for the ``-p`` one-shot path
  stops printing assistant text to stdout on turn complete.

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

# Model + harness are hardcoded because the test name advertises
# "codex harness"; a per-harness characterization test is
# meaningless without pinning the harness under test.
# databricks-gpt-5-4-mini is the gateway model documented in the
# repo-level CLAUDE.md as the safe default for OpenAI-flavored
# harnesses (codex is an OpenAI-native coding agent).
_MODEL = resolve_model("databricks-gpt-5-4-mini", key=__name__)
_HARNESS = "codex"
_PROMPT = "say hi in 5 words"

# Minimum assistant-text length. Anything longer than "hi" proves
# the turn produced a genuine model reply (not an empty response
# or a pure error banner from the codex app-server).
_MIN_ASSISTANT_CHARS = 4

# Subprocess timeout. codex boots its app-server subprocess and
# establishes the JSON-RPC stream before the first turn event,
# so it's slower than openai-agents; 180s matches claude-sdk's
# headroom and keeps pace with cold-start latency on CI hosts.
_RUN_TIMEOUT_SEC = 180


@pytest.fixture
def codex_available() -> bool:
    """
    Availability probe for the codex harness prerequisites.

    ``CodexExecutor`` shells out to the ``codex`` CLI binary
    (installed via ``npm i -g @openai/codex`` typically).
    Without it the executor raises immediately on session start.
    CI environments commonly lack the binary.

    :returns: True when ``codex`` is on PATH.
    """
    return which("codex") is not None


def test_per_harness_codex_one_shot(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    patched_databrickscfg: None,
    codex_available: bool,
) -> None:
    """
    ``omnigent run hello_world.yaml --harness codex -p <prompt>``
    exits 0 and emits a non-trivial assistant reply.

    Uses ``patched_databrickscfg`` because ``CodexExecutor`` routes
    model calls through a Databricks-specific config override set
    constructed from ``~/.databrickscfg`` via
    ``_read_databrickscfg`` — OAuth-profile tokens silently 403
    the codex app-server's model requests. This matches the
    claude-sdk pattern; the workaround is documented in
    ``OMNIGENT_INTEGRATION.md`` as the pre-phase-1 baseline and
    disappears once the ``databricks-sdk`` rewrite lands.

    :param omnigent_python: Interpreter with omnigent
        installed and importable.
    :param omnigent_repo_root: Cwd for the subprocess so the
        YAML spec and example tool modules resolve on sys.path.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` populated from
        ``--llm-api-key``.
    :param patched_databrickscfg: Fixture that rewrites
        ``~/.databrickscfg`` to PAT form for the test and
        restores it on teardown.
    :param codex_available: True when the ``codex`` CLI is
        present. On False the test fails with an explicit reason
        per CLAUDE.md rule 30 (no silent skips).
    """
    if not codex_available:
        pytest.fail(
            "codex harness prerequisite missing: the 'codex' CLI "
            "binary must be installed on PATH (install via "
            "'npm i -g @openai/codex')."
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

    # codex's app-server can print its own one-line startup
    # diagnostics to stderr before the first turn event. Known
    # benign lines (e.g. the ``App server listening`` banner) are
    # excluded before the cleanliness assertion so the test
    # doesn't spuriously fail on harmless informational output.
    stderr_stripped = "\n".join(
        line
        for line in result.stderr.splitlines()
        # Codex prints a Node runtime deprecation line on some
        # Node installs; orthogonal to the behavior under test.
        if "DeprecationWarning" not in line and "App server listening" not in line
    ).strip()

    observed: dict[str, Any] = {
        "exit_code": result.returncode,
        "stderr_is_clean": stderr_stripped == "",
        # Trimmed because whitespace around LLM output is noisy
        # and not something we want the snapshot comparator to
        # trip on.
        "assistant_text": result.stdout.strip(),
    }

    # Full stderr surfaced on failure so CI logs show WHY the run
    # went wrong (e.g. 403 auth, missing binary) — stderr here is
    # opaque unless we dump it in the failure message.
    diffs = compare_snapshot("test_per_harness_codex", observed)
    assert diffs == [], (
        "Snapshot mismatch for codex run:\n"
        + "\n".join(diffs)
        + f"\n\nstdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )
    # Separate assertion so a length regression names the length
    # check directly instead of being buried in the snapshot diff.
    assert len(observed["assistant_text"]) >= _MIN_ASSISTANT_CHARS, (
        f"Codex assistant text shorter than {_MIN_ASSISTANT_CHARS} "
        f"chars; got {observed['assistant_text']!r}"
    )
