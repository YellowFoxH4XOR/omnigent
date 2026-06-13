"""
REPL approval-flow e2e test — sessions API variant.

Sessions-API parallel of ``test_repl_approval_e2e.py``. Spawns
``omnigent run <yaml>`` (sessions API is default; the Databricks
profile rides on the config-home auth block)
under pexpect and drives the same approval CUJs through the
``/v1/sessions`` path instead of ``/v1/responses``.

The agent fixtures are the same as the legacy tests (ask-demo,
e2e-tool-gate, etc.) — only the spawn command differs. Each
test asserts the identical behavior: approval prompts surface,
verdicts route through the sessions event path, and the LLM
reply (or deny sentinel) renders correctly.

Prerequisites:
    - ``pexpect`` installed (4.9+).
    - A Databricks profile authenticated (default ``dev``);
      override via ``OMNIGENT_SESSIONS_E2E_PROFILE=<name>``.
    - Gated under CI unless ``OMNIGENT_RUN_LIVE_REPL_E2E=1``.

Usage::

    python -m pytest tests/e2e/test_repl_sessions_approval_e2e.py -v
"""

from __future__ import annotations

import configparser
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests._model_pools import resolve_model

pexpect = pytest.importorskip("pexpect")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_ENV_VAR = "OMNIGENT_SESSIONS_E2E_PROFILE"
_DEFAULT_PROFILE = "dev"
_MODEL = resolve_model("databricks-claude-sonnet-4-6", key=__name__)
_ASK_DEMO_YAML = _REPO_ROOT / "tests" / "resources" / "agents" / "ask-demo" / "ask-demo.yaml"
_FIXTURES_DIR = _REPO_ROOT / "tests" / "_fixtures" / "agents"
_TOOL_GATE_DIR = _FIXTURES_DIR / "e2e-tool-gate"
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences before substring search."""
    return _ANSI_RE.sub("", text)


def _resolve_profile_token(profile: str) -> str | None:
    """Resolve a Databricks bearer for ``profile`` via the CLI."""
    cli = shutil.which("databricks")
    if cli is None:
        return None
    try:
        proc = subprocess.run(
            [cli, "auth", "token", "--profile", profile],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    token = payload.get("access_token")
    return token if isinstance(token, str) and token else None


def _resolve_workspace_host(profile: str) -> str | None:
    """Read ``~/.databrickscfg``'s ``host`` for ``profile``."""
    cfg_path = Path.home() / ".databrickscfg"
    if not cfg_path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(cfg_path)
    except configparser.Error:
        return None
    if profile not in parser:
        return None
    host = parser[profile].get("host")
    return host.rstrip("/") if host else None


@pytest.fixture(scope="module")
def sessions_credentials() -> dict[str, str]:
    """Resolve Databricks credentials for the sessions e2e profile."""
    if os.environ.get("CI", "").lower() == "true" and not os.environ.get(
        "OMNIGENT_RUN_LIVE_REPL_E2E"
    ):
        pytest.skip(
            "live REPL e2e gated under CI; set OMNIGENT_RUN_LIVE_REPL_E2E=1 to opt in.",
        )
    profile = os.environ.get(_PROFILE_ENV_VAR, _DEFAULT_PROFILE)
    token = _resolve_profile_token(profile)
    if token is None:
        pytest.skip(
            f"Databricks profile {profile!r} not authenticated; "
            f"run `databricks auth login --profile {profile}` first.",
        )
    host = _resolve_workspace_host(profile)
    if host is None:
        pytest.skip(
            f"Databricks profile {profile!r} is missing a 'host' entry.",
        )
    return {"profile": profile, "token": token, "host": host}


def _build_repl_env(creds: dict[str, str]) -> dict[str, str]:
    """Build the pexpect environment dict for REPL spawning.

    Centralises SDK PYTHONPATH injection, credential plumbing,
    and the suppression of variables that leak from the outer
    agent process.

    Databricks routing for the spawned CLI comes from the global
    config's ``auth:`` block in an isolated ``OMNIGENT_CONFIG_HOME``
    (the supported replacement for the removed ``--profile`` flag).
    """
    from tests.e2e.omnigent._pexpect_harness import ensure_repl_test_theme_env

    sdk_paths = [
        str(_REPO_ROOT / "sdks" / "python-client"),
        str(_REPO_ROOT / "sdks" / "ui"),
    ]
    existing_pp = os.environ.get("PYTHONPATH", "")
    merged_pp = (
        os.pathsep.join([*sdk_paths, existing_pp]) if existing_pp else os.pathsep.join(sdk_paths)
    )

    config_home = Path(tempfile.mkdtemp(prefix="omnigent-approval-config-"))
    (config_home / "config.yaml").write_text(
        f"auth:\n  type: databricks\n  profile: {creds['profile']}\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "DATABRICKS_TOKEN": creds["token"],
        "OPENAI_API_KEY": creds["token"],
        "OPENAI_BASE_URL": f"{creds['host']}/serving-endpoints",
        "DATABRICKS_CONFIG_PROFILE": creds["profile"],
        "OMNIGENT_CONFIG_HOME": str(config_home),
        "PYTHONPATH": merged_pp,
        "TERM": "xterm-256color",
        "LINES": "40",
        "COLUMNS": "120",
        "PROMPT_TOOLKIT_NO_CPR": "1",
    }
    for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE", "CODEX"):
        env.pop(k, None)
    return ensure_repl_test_theme_env(env)


def _spawn_sessions_repl(
    yaml_path: Path,
    creds: dict[str, str],
    *,
    timeout: int = 120,
) -> Any:
    """
    Spawn ``omnigent run`` under a PTY (sessions API is default).

    :param yaml_path: Path to the agent YAML (can be a directory
        containing ``config.yaml`` or a standalone ``.yaml``).
    :param creds: Dict with ``profile``, ``token``, ``host``.
    :param timeout: pexpect timeout in seconds.
    :returns: The pexpect child process.
    """
    return pexpect.spawn(
        sys.executable,
        [
            "-m",
            "omnigent",
            "run",
            str(yaml_path),
            "--no-session",
            "--model",
            _MODEL,
        ],
        env=_build_repl_env(creds),
        cwd=str(_REPO_ROOT),
        encoding="utf-8",
        codec_errors="replace",
        timeout=timeout,
        dimensions=(40, 120),
    )


def _spawn_repl_with_args(
    yaml_path: Path,
    creds: dict[str, str],
    *,
    extra_args: list[str] | None = None,
    timeout: int = 120,
) -> Any:
    """Spawn ``omnigent run`` with caller-supplied CLI args.

    Unlike :func:`_spawn_sessions_repl` this does NOT inject
    Sessions API is the default; the caller controls which
    flags are passed via *extra_args*.

    :param yaml_path: Path to the agent YAML.
    :param creds: Dict with ``profile``, ``token``, ``host``.
    :param extra_args: Additional CLI flags, e.g. ``["--log"]``.
    :param timeout: pexpect timeout in seconds.
    :returns: The pexpect child process.
    """
    args = [
        "-m",
        "omnigent",
        "run",
        str(yaml_path),
        "--no-session",
        "--model",
        _MODEL,
    ]
    if extra_args:
        args.extend(extra_args)
    return pexpect.spawn(
        sys.executable,
        args,
        env=_build_repl_env(creds),
        cwd=str(_REPO_ROOT),
        encoding="utf-8",
        codec_errors="replace",
        timeout=timeout,
        dimensions=(40, 120),
    )


def _wait_for_prompt_ready(child: Any, timeout: float = 60.0) -> None:
    """Wait for the REPL prompt (``❯``) to appear."""
    child.expect("❯", timeout=timeout)


def _read_pending(child: Any, seconds: float = 0.3) -> str:
    """Non-blocking read of buffered output, ANSI-stripped."""
    with contextlib.suppress(pexpect.EOF):
        child.expect(pexpect.TIMEOUT, timeout=seconds)
    captured = child.before or ""
    if isinstance(captured, bytes):
        captured = captured.decode("utf-8", errors="replace")
    return _strip_ansi(captured)


def _clean_exit(child: Any) -> None:
    """Best-effort clean exit of the REPL."""
    try:
        child.sendcontrol("d")
        child.expect(pexpect.EOF, timeout=10)
    except pexpect.ExceptionPexpect:
        pass
    if child.isalive():
        child.terminate(force=True)


# ── CUJ 1: Single approval allows LLM response ─────────


def test_sessions_single_approval_allows_llm_response(
    sessions_credentials: dict[str, str],
) -> None:
    """
    Sessions API variant: approval prompt surfaces, user types
    ``y``, LLM reply renders.
    """
    child = _spawn_sessions_repl(_ASK_DEMO_YAML, sessions_credentials)
    try:
        _wait_for_prompt_ready(child)
        child.send("Hello\r")
        child.expect("approval required", timeout=30)
        child.send("y\r")
        child.expect("approved", timeout=10)

        buffered = _read_pending(child, seconds=5.0)
        buffered += _read_pending(child, seconds=3.0)
        assert re.search(r"[A-Za-z]{3,}", buffered), (
            f"No LLM response after approval.\nBuffer:\n{buffered[:800]}"
        )
    except pexpect.EOF:
        # Dump full buffer for diagnosis.
        buf = _strip_ansi(child.before or "")
        pytest.fail(f"REPL exited early. Full buffer:\n{buf[-2000:]}")
    finally:
        _clean_exit(child)


# ── CUJ 2: Refusal shows deny sentinel ──────────────────


def test_sessions_refusal_shows_deny_sentinel(
    sessions_credentials: dict[str, str],
) -> None:
    """
    Sessions API variant: user refuses → deny sentinel appears.
    """
    child = _spawn_sessions_repl(_ASK_DEMO_YAML, sessions_credentials)
    try:
        _wait_for_prompt_ready(child)
        child.send("Hello\r")
        child.expect("approval required", timeout=30)
        child.send("n\r")
        child.expect("refused", timeout=10)

        buffered = _read_pending(child, seconds=5.0)
        assert "DENIED" in buffered.upper() or "refused" in buffered.lower(), (
            f"No deny sentinel after refusal.\nBuffer:\n{buffered[:800]}"
        )
    finally:
        _clean_exit(child)


# ── CUJ 3: Multi-turn fires approval each turn ──────────


def test_sessions_two_turns_fires_one_approval_per_turn(
    sessions_credentials: dict[str, str],
) -> None:
    """
    Sessions API variant: each turn produces exactly one
    approval prompt.
    """
    child = _spawn_sessions_repl(_ASK_DEMO_YAML, sessions_credentials)
    try:
        _wait_for_prompt_ready(child)

        # Turn 1.
        child.send("First message\r")
        child.expect("approval required", timeout=30)
        child.send("y\r")
        child.expect("approved", timeout=10)
        _read_pending(child, seconds=5.0)

        # Turn 2.
        child.send("Second message\r")
        child.expect("approval required", timeout=30)
        child.send("y\r")
        child.expect("approved", timeout=10)
        buffered = _read_pending(child, seconds=5.0)
        assert re.search(r"[A-Za-z]{3,}", buffered), (
            f"No reply after second-turn approval.\nBuffer:\n{buffered[:800]}"
        )
    finally:
        _clean_exit(child)


# ── CUJ 4: Approve-always caches for session ────────────


def test_sessions_approve_always_caches_for_later_turns(
    sessions_credentials: dict[str, str],
) -> None:
    """
    Sessions API variant: ``a`` (approve always) on first turn
    suppresses the prompt on the second turn.
    """
    child = _spawn_sessions_repl(_ASK_DEMO_YAML, sessions_credentials)
    try:
        _wait_for_prompt_ready(child)

        # Turn 1: approve always.
        child.send("First\r")
        child.expect("approval required", timeout=30)
        child.send("a\r")
        child.expect("approved always", timeout=10)
        _read_pending(child, seconds=5.0)

        # Turn 2: should auto-approve (no prompt).
        child.send("Second\r")
        buffered = _read_pending(child, seconds=8.0)
        approval_count = buffered.count("approval required")
        assert approval_count == 0, (
            f"Approval prompt appeared after approve-always. "
            f"Count={approval_count}\nBuffer:\n{buffered[:800]}"
        )
        assert re.search(r"[A-Za-z]{3,}", buffered), (
            f"No auto-approved reply.\nBuffer:\n{buffered[:800]}"
        )
    finally:
        _clean_exit(child)


# ── CUJ 5: Tool call approval ───────────────────────────


def test_sessions_tool_call_approval_allows_tool(
    sessions_credentials: dict[str, str],
) -> None:
    """
    Sessions API variant: tool-phase approval surfaces,
    user approves, tool runs.
    """
    tool_gate_yaml = _TOOL_GATE_DIR / "e2e-tool-gate.yaml"
    if not tool_gate_yaml.exists():
        pytest.skip(f"Fixture {tool_gate_yaml} not found")
    child = _spawn_sessions_repl(tool_gate_yaml, sessions_credentials)
    try:
        _wait_for_prompt_ready(child, timeout=60)
        child.send("Use the tool\r")
        child.expect("approval required", timeout=30)
        child.send("y\r")
        child.expect("approved", timeout=10)
        buffered = _read_pending(child, seconds=8.0)
        assert re.search(r"[A-Za-z]{3,}", buffered), (
            f"No response after tool approval.\nBuffer:\n{buffered[:800]}"
        )
    finally:
        _clean_exit(child)


# ── CUJ 6: Default flag uses sessions API ─────────────────


def _write_simple_agent_yaml(directory: Path) -> Path:
    """Write a minimal agent YAML with no policies (no approval).

    Returns the path to the created YAML file.
    """
    yaml_path = directory / "simple_hello.yaml"
    yaml_path.write_text(
        "name: simple_hello\n"
        "prompt: >-\n"
        "  You are a friendly assistant. Respond in exactly one short sentence.\n",
    )
    return yaml_path


def test_sessions_default_flag_works(
    sessions_credentials: dict[str, str],
    tmp_path: Path,
) -> None:
    """
    Spawns the REPL through the default sessions path and verifies
    the agent responds through the sessions API.

    A simple agent with no policies is used so the test is not blocked
    by an approval prompt -- we only need to confirm the sessions
    default works end-to-end.
    """
    yaml_path = _write_simple_agent_yaml(tmp_path)
    child = _spawn_repl_with_args(yaml_path, sessions_credentials)
    try:
        _wait_for_prompt_ready(child, timeout=60)
        child.send("Say hello in exactly five words\r")

        # Wait for the LLM to respond.  The agent has no policies so
        # there should be no approval gate -- text should arrive.
        buffered = _read_pending(child, seconds=10.0)
        buffered += _read_pending(child, seconds=5.0)

        # The LLM must produce at least one word of real text.
        # If the sessions path were broken, the REPL would either
        # error at startup or render no assistant output.
        assert re.search(r"[A-Za-z]{3,}", buffered), (
            f"No LLM response rendered -- sessions-API default may "
            f"not be active.\nBuffer:\n{buffered[:800]}"
        )
    except pexpect.EOF:
        buf = _strip_ansi(child.before or "")
        pytest.fail(f"REPL exited early (default sessions flag). Full buffer:\n{buf[-2000:]}")
    finally:
        _clean_exit(child)
