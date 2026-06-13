"""Phase 0 characterization test — ``hello_world.yaml`` end-to-end.

Runs the truly-minimal ``examples/hello_world.yaml`` spec (no
``executor`` block, no ``tools`` block, no ``policies``) through
``omnigent run`` with a model override and snapshots structural
observations.

This complements the sibling ``test_yaml_hello_world.py`` file,
which — despite its name — exercises ``agent_with_tools.yaml`` to
cover tool-call plumbing. That file will be renamed at merge
time; this file is the literal ``hello_world.yaml`` coverage the
Phase 0 design calls out.

**What breaks if this fails:**
- Omnigent' YAML spec parser regresses on the minimal
  ``name:`` + ``prompt:`` shape (no executor, no tools, no
  policies).
- ``omnigent.loader`` stops applying CLI ``--model`` as a
  fallback when the YAML omits ``executor.model``.
- The default harness selection path regresses (openai-agents is
  the default when the YAML has no executor block).
- ``omnigent.cli._run_agent`` for the ``-p`` one-shot path
  stops printing the assistant text on turn complete.

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
YAML→agent characterization.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Any

import pytest

from tests.e2e._harness_probes import HARNESS_HARNESS_MODELS, HARNESS_IDS
from tests.e2e.omnigent._snapshot import compare_snapshot

_PROMPT = "say hi in 5 words"

# Minimum assistant-text length. Anything longer than "hi" proves
# the turn produced a real model reply, not an empty response or
# a pure error banner.
_MIN_ASSISTANT_CHARS = 4

# Subprocess timeout matches the other per-harness tests — 180s
# headroom for cold starts on loaded CI hosts.
_RUN_TIMEOUT_SEC = 180


def _check_harness_available(harness: str, omnigent_python: Path) -> None:
    """
    Fail loud if the parametrized harness's prerequisites are missing.

    Mirrors the per-harness availability checks in
    ``test_per_harness_claude_sdk.py`` and
    ``test_per_harness_codex.py``. Following CLAUDE.md rule 30
    we fail rather than silently skip so missing prerequisites
    stay visible.

    :param harness: The harness identifier under test.
    :param omnigent_python: The subprocess interpreter — used
        to probe Python-package availability.
    """
    if harness == "claude-sdk":
        probe = subprocess.run(
            [
                str(omnigent_python),
                "-c",
                "import importlib.util, sys; "
                "sys.exit(0 if importlib.util.find_spec('claude_agent_sdk') else 1)",
            ],
            capture_output=True,
        )
        if probe.returncode != 0 or which("claude") is None:
            pytest.fail(
                "claude-sdk harness prerequisites missing: both the "
                "'claude_agent_sdk' Python package and the 'claude' CLI "
                "binary must be present on PATH."
            )
    elif harness == "codex":
        if which("codex") is None:
            pytest.fail(
                "codex harness prerequisite missing: the 'codex' CLI "
                "binary must be installed on PATH (install via "
                "'npm i -g @openai/codex')."
            )


@pytest.mark.parametrize("harness,model", HARNESS_HARNESS_MODELS, ids=HARNESS_IDS)
def test_yaml_hello_world_real(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    patched_databrickscfg: None,
    harness: str,
    model: str,
) -> None:
    """
    ``omnigent run hello_world.yaml --harness <harness> --model
    <model> -p <prompt>`` exits 0 and emits a non-trivial
    assistant reply.

    The YAML is deliberately minimal (name + prompt only) — this
    test proves the harness + CLI-model-override pipeline
    still works for the simplest valid spec. If this fails, the
    "getting started" experience is broken. Parametrized so each
    wrapped harness exercises the path.

    :param omnigent_python: Interpreter with omnigent
        installed and importable.
    :param omnigent_repo_root: Cwd for the subprocess so the
        YAML path resolves and any relative example imports work.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` populated from
        ``--llm-api-key``.
    :param patched_databrickscfg: Rewrites the dogfood profile
        in ``~/.databrickscfg`` to PAT form for the test (claude
        and codex harnesses both read the file directly and
        OAuth profiles 403).
    :param harness: The harness identifier from
        :data:`HARNESS_HARNESS_MODELS`.
    :param model: The harness-routed model identifier from
        :data:`HARNESS_HARNESS_MODELS`.
    """
    _check_harness_available(harness, omnigent_python)
    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "hello_world.yaml"

    result = subprocess.run(
        [
            str(omnigent_python),
            "-m",
            "omnigent",
            "run",
            str(yaml_path),
            "--model",
            model,
            "--harness",
            harness,
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
    diffs = compare_snapshot("test_yaml_hello_world_real", observed)
    assert diffs == [], (
        "Snapshot mismatch for hello_world.yaml run:\n"
        + "\n".join(diffs)
        + f"\n\nstdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )
    assert len(observed["assistant_text"]) >= _MIN_ASSISTANT_CHARS, (
        f"hello_world assistant text shorter than "
        f"{_MIN_ASSISTANT_CHARS} chars; got "
        f"{observed['assistant_text']!r}"
    )
