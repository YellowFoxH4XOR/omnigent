"""Phase 0 characterization test — YAML-driven agent with tools.

Despite the file name (kept for parity with the design-doc test
catalog), this test targets ``agent_with_tools.yaml`` because
Phase 0's YAML characterization requires exercising a tool-bearing
agent — ``hello_world.yaml`` has no ``tools:`` block and
therefore can't prove tool-call plumbing. The design explicitly
calls out ``agent_with_tools.yaml`` as the representative
tool-bearing fixture for this slot.

**What breaks if this fails:**
- Omnigent' YAML spec parser regresses on ``tools.*`` entries
  (``function`` / ``cancellable_function`` types).
- The wrapped harness loses its MCP tool bridging or its
  prompt-construction path.
- Per-YAML defaults fail to pick up the ``callable:`` dotted
  paths via ``importlib.import_module`` — the tool never gets
  registered and the agent can't invoke it.
- ``omnigent.cli`` one-shot path stops streaming tool-call
  lifecycle lines (``◦ <tool>`` / ``• <tool>``) to stdout.

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

_PROMPT = "What is 3 + 4? Use the calculate tool."

# ``agent_with_tools.yaml`` defines a ``calculate`` tool. The
# REPL's tool-lifecycle lines look like
# ``◦ calculate`` (start) and ``• calculate (NNms)`` (complete).
# We snapshot the substring ``"calculate"`` so the comparator
# succeeds as long as either line appears, regardless of the
# exact timing format.
_EXPECTED_TOOL_NAME = "calculate"

# Minimum assistant-text length. The prompt asks a direct
# arithmetic question so the reply is typically short but must
# be longer than e.g. "7" to prove the full turn streamed.
_MIN_ASSISTANT_CHARS = 3

_RUN_TIMEOUT_SEC = 180


def _check_harness_available(harness: str, omnigent_python: Path) -> None:
    """
    Fail loud if the parametrized harness's prerequisites are missing.

    Mirrors the per-harness availability checks in
    ``test_per_harness_claude_sdk.py`` and
    ``test_per_harness_codex.py``. claude-sdk needs the SDK
    package + the ``claude`` CLI binary; codex needs the
    ``codex`` CLI binary. Following CLAUDE.md rule 30 we fail
    instead of silently skipping so missing prerequisites stay
    visible.

    :param harness: The harness identifier under test.
    :param omnigent_python: The subprocess interpreter — used
        to probe Python-package availability, not the running
        test interpreter.
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
def test_yaml_agent_with_tools(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    patched_databrickscfg: None,
    harness: str,
    model: str,
) -> None:
    """
    Running ``omnigent run agent_with_tools.yaml --harness
    <harness> -p <calc-prompt>`` completes cleanly and the
    ``calculate`` tool appears in stdout.

    Parametrized across every wrapped harness (claude-sdk +
    codex) so the YAML→tools pipeline is verified end-to-end
    once per harness.

    :param omnigent_python: Interpreter with omnigent +
        the harness's SDK installed.
    :param omnigent_repo_root: Cwd for the subprocess — the
        YAML's ``callable:`` entries (``tests.resources.examples._shared.tool_functions
        .calculate``) only import if the repo root is on
        sys.path, which ``cwd=...`` achieves.
    :param omnigent_credentials_env: Env vars with
        ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
        ``DATABRICKS_CONFIG_PROFILE`` populated from
        ``--llm-api-key``.
    :param patched_databrickscfg: Rewrites the dogfood profile
        in ``~/.databrickscfg`` to PAT form for the test's
        duration (claude-sdk and codex both read the file
        directly and OAuth profiles 403).
    :param harness: The harness identifier from
        :data:`HARNESS_HARNESS_MODELS`.
    :param model: The harness-routed model identifier from
        :data:`HARNESS_HARNESS_MODELS`.
    """
    _check_harness_available(harness, omnigent_python)
    yaml_path = omnigent_repo_root / "tests" / "resources" / "examples" / "agent_with_tools.yaml"

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
        # Combined stdout because the tool-lifecycle lines
        # (``◦ calculate`` / ``• calculate``) and the assistant
        # reply both land on stdout, not stderr. The snapshot's
        # ``contains`` comparator checks for the tool name.
        "stdout": result.stdout,
        "stderr_is_clean": result.stderr.strip() == "",
    }

    diffs = compare_snapshot("test_yaml_hello_world", observed)
    assert diffs == [], (
        "Snapshot mismatch for agent_with_tools run:\n"
        + "\n".join(diffs)
        + f"\n\nstdout:\n{result.stdout!r}\n\nstderr:\n{result.stderr!r}"
    )

    # Length check on the *assistant* portion, not the
    # tool-lifecycle lines. We strip ANSI + the known banner
    # prefix to isolate the reply.
    stripped = _strip_tool_chatter(result.stdout)
    assert len(stripped) >= _MIN_ASSISTANT_CHARS, (
        f"Assistant text shorter than {_MIN_ASSISTANT_CHARS} "
        f"chars after stripping tool lifecycle lines; got "
        f"{stripped!r} (full stdout: {result.stdout!r})"
    )
    # Belt-and-braces — the snapshot's ``contains`` comparator
    # already covers this, but naming the assertion explicitly
    # makes the failure message self-explanatory if the
    # snapshot file is ever accidentally deleted.
    assert _EXPECTED_TOOL_NAME in result.stdout, (
        f"Expected tool name {_EXPECTED_TOOL_NAME!r} not found "
        f"in stdout; the {harness} harness did not invoke "
        f"the calculate tool.\n\nstdout:\n{result.stdout!r}"
    )


def _strip_tool_chatter(stdout: str) -> str:
    """
    Remove known tool-lifecycle marker lines from stdout.

    The omnigent CLI prints ``◦ <tool>`` (queued) and
    ``• <tool> (NNms)`` (done) lines around tool calls regardless
    of which harness fired them. For the assistant-length
    assertion we want to measure only the natural-language reply,
    not those markers.

    :param stdout: Raw stdout from ``omnigent run``.
    :returns: The stdout with tool lifecycle lines removed,
        trimmed of leading/trailing whitespace.
    """
    kept: list[str] = []
    for line in stdout.splitlines():
        stripped_line = line.strip()
        # Both markers use exotic unicode glyphs not likely to
        # appear in an arithmetic reply, so prefix-matching
        # them is safe.
        if stripped_line.startswith(("◦ ", "• ")):
            continue
        kept.append(line)
    return "\n".join(kept).strip()
