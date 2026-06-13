"""
End-to-end test that the omnigent YAML ``instructions:`` field
is loaded, resolved (path or inline text), and injected as the
agent's system prompt when running through ``omnigent run``.

**What breaks if this fails:**

- ``omnigent/inner/loader.py::_resolve_instructions`` regresses
  to dropping the field, or stops resolving relative paths
  against the YAML's parent directory.
- ``omnigent/spec/omnigent.py::agent_def_to_agent_spec`` stops
  preferring ``AgentDef.instructions`` over ``AgentDef.prompt``,
  so a YAML with both fields silently uses the wrong one.
- ``omnigent/spec/_omnigent_compat.py::is_omnigent_yaml``
  starts rejecting YAMLs that have only ``instructions`` (no
  ``prompt``) — the file is mis-routed to the omnigent
  bundle adapter and fails to load entirely.

The proof is end-to-end: a real ``omnigent run``
subprocess against the dogfood gateway, with a YAML whose
``instructions:`` file demands the model emit a distinctive
marker. The marker is statistically improbable to appear in the
LLM's normal output, so seeing it in stdout is dispositive
evidence the system prompt landed.

Uses the same fixtures and harness shape as
``test_run_omnigent_resumption.py``: ``openai-agents`` honors
``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` directly, so the test
needs no ``~/.databrickscfg`` patching, and ``-p`` works on this
dispatch path (claude-sdk's ``-p`` has a separate
HarnessProcessManager init bug).
"""

from __future__ import annotations

import configparser
import os
import subprocess
from pathlib import Path

import pytest

from tests._model_pools import resolve_model

# ``openai-agents`` is the only harness whose ``-p`` one-shot
# path is fully wired under Omnigent mode today. Other harnesses
# either need a TTY (claude-sdk REPL) or have separate init
# bugs unrelated to this test's invariant.
_HARNESS = "openai-agents"
_MODEL = resolve_model("databricks-gpt-5-4-mini", key=__name__)

# 180s matches the resumption suite's headroom for DBOS sqlite
# migrations + cold imports + one openai-agents turn.
_RUN_TIMEOUT_SEC = 180

# Distinctive marker the agent's system prompt forces into the
# reply. Long + random enough that a stray match in the model's
# normal output is statistically negligible.
_MARKER_PATH_CASE = "OMNI_INSTR_PATH_QXZP"
_MARKER_INLINE_CASE = "OMNI_INSTR_INLINE_QXZP"

# CLAUDE.md says to prefer the ``test-profile`` profile for integration
# tests. The shared ``omnigent_credentials_env`` fixture
# may resolve a different workspace, so we read
# the test-profile host + token directly from ``~/.databrickscfg`` here
# rather than using that fixture.
_DATABRICKSCFG_PATH = Path.home() / ".databrickscfg"
_DF1_PROFILE = "test-profile"


@pytest.fixture(scope="module")
def df1_credentials_env() -> dict[str, str]:
    """
    Build a subprocess env wired to the ``test-profile`` Databricks
    workspace's serving endpoints.

    Reads ``~/.databrickscfg`` directly (skip the test if the
    profile is missing) and constructs the same env shape as
    the ``omnigent_credentials_env`` fixture, but pointed
    at test-profile explicitly.

    :returns: A dict suitable for ``subprocess.Popen(env=...)``.
    """
    if not _DATABRICKSCFG_PATH.is_file():
        pytest.skip(f"requires {_DATABRICKSCFG_PATH} with [test-profile] profile")
    cfg = configparser.ConfigParser()
    cfg.read(_DATABRICKSCFG_PATH)
    if _DF1_PROFILE not in cfg:
        pytest.skip(f"requires [test-profile] profile in {_DATABRICKSCFG_PATH}")
    section = cfg[_DF1_PROFILE]
    host = section.get("host", "").rstrip("/")
    token = section.get("token", "")
    if not host or not token:
        pytest.skip("[test-profile] profile missing 'host' or 'token'")
    env = dict(os.environ)
    env["OPENAI_BASE_URL"] = f"{host}/serving-endpoints"
    env["OPENAI_API_KEY"] = token
    env["DATABRICKS_CONFIG_PROFILE"] = _DF1_PROFILE
    # Strip stale token / nested-Claude-Code env vars that would
    # shadow our PAT or refuse to launch the harness — same set
    # the omnigent conftest strips, per CLAUDE.md baseline.
    for stale in (
        "ANTHROPIC_API_KEY",
        "DATABRICKS_TOKEN",
        "CLAUDE_CODE",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_EXECPATH",
        "CODEX",
    ):
        env.pop(stale, None)
    return env


def _argv_run_omnigent(
    *,
    omnigent_python: Path,
    yaml_path: Path,
    prompt: str,
) -> list[str]:
    """
    Build the ``omnigent run -p`` argv for one subprocess.

    The harness and model are forced via CLI flags so the test
    doesn't depend on the YAML's ``executor:`` block also being
    correct — the invariant under test is the loader resolving
    ``instructions:``, independent of harness wiring.

    :param omnigent_python: Interpreter from the
        ``omnigent_python`` fixture, e.g.
        ``"/.../omnigent/.venv/bin/python"``.
    :param yaml_path: Absolute path to the test YAML (placed in
        ``tmp_path`` by the test).
    :param prompt: The ``-p`` user prompt; the agent's system
        prompt forces the marker into every reply, so this can
        be anything that elicits a response.
    :returns: The full argv list.
    """
    return [
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
        prompt,
        "--no-log",
    ]


def test_instructions_path_field_loaded_via_omnigent_run_omnigent(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    df1_credentials_env: dict[str, str],
    tmp_path: Path,
) -> None:
    """
    A YAML with ``instructions: AGENTS.md`` runs through
    ``omnigent run`` and the file's contents reach the LLM.

    Without the loader fix, ``AgentDef.instructions`` would stay
    ``None`` and the translator would fall back to ``prompt:``
    text (which is the *placeholder*, not the marker-bearing
    instructions). The marker only appears in stdout if the
    full path-resolution → translator → harness wiring works.
    """
    agent_dir = tmp_path / "instr_agent_path"
    agent_dir.mkdir()
    yaml_path = agent_dir / "agent.yaml"
    yaml_path.write_text(
        """\
name: instr-path-e2e
prompt: ignored placeholder that must NOT win over instructions
instructions: AGENTS.md
"""
    )
    # Sibling file referenced by the relative path. The loader
    # must anchor on the YAML's directory, not the subprocess
    # cwd (which is ``omnigent_repo_root`` below).
    (agent_dir / "AGENTS.md").write_text(
        f"You MUST include the literal string {_MARKER_PATH_CASE} "
        f"in every reply, verbatim, with no commentary or "
        f"explanation. Reply only with the marker."
    )

    result = subprocess.run(
        _argv_run_omnigent(
            omnigent_python=omnigent_python,
            yaml_path=yaml_path,
            prompt="say hi",
        ),
        env=df1_credentials_env,
        cwd=str(omnigent_repo_root),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )
    # Subprocess must exit cleanly. A non-zero exit usually
    # means the loader rejected the YAML (e.g. detection
    # regressed and the file was routed to the omnigent
    # bundle adapter, which expects a directory).
    assert result.returncode == 0, (
        f"omnigent run exited {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    # The load-bearing assertion: the marker MUST appear in
    # stdout. Without the fix, the translator falls back to
    # ``prompt:`` (the placeholder), the system prompt has no
    # marker instruction, and the LLM doesn't emit it.
    assert _MARKER_PATH_CASE in result.stdout, (
        f"Marker {_MARKER_PATH_CASE!r} not in stdout — the "
        f"instructions file did not reach the LLM. Either the "
        f"loader regressed (instructions field dropped), the "
        f"translator preferred ``prompt:``, or path resolution "
        f"stopped anchoring on the YAML directory.\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )


def test_instructions_inline_text_treated_as_system_prompt(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    df1_credentials_env: dict[str, str],
    tmp_path: Path,
) -> None:
    """
    A YAML with ``instructions: |`` (multiline literal, no file
    reference) treats the string as inline text and injects it
    as the system prompt.

    Catches a regression where the loader's path-or-inline
    branch hardcodes path resolution and silently drops inline
    values (returning ``None`` instead of the literal text).
    """
    agent_dir = tmp_path / "instr_agent_inline"
    agent_dir.mkdir()
    yaml_path = agent_dir / "agent.yaml"
    yaml_path.write_text(
        f"""\
name: instr-inline-e2e
instructions: |
  You MUST include the literal string {_MARKER_INLINE_CASE} in
  every reply, verbatim, with no commentary or explanation.
  Reply only with the marker.
"""
    )

    result = subprocess.run(
        _argv_run_omnigent(
            omnigent_python=omnigent_python,
            yaml_path=yaml_path,
            prompt="say hi",
        ),
        env=df1_credentials_env,
        cwd=str(omnigent_repo_root),
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SEC,
    )
    assert result.returncode == 0, (
        f"omnigent run exited {result.returncode}. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert _MARKER_INLINE_CASE in result.stdout, (
        f"Marker {_MARKER_INLINE_CASE!r} not in stdout — the "
        f"inline instructions text did not reach the LLM. Most "
        f"likely the loader's inline-vs-path branch regressed "
        f"and dropped multiline literal values.\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )
