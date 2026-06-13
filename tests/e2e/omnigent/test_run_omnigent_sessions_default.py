"""End-to-end test: ``omnigent run`` renders through sessions.

Smoke regression for the default sessions REPL path. The
unit tests in ``tests/repl/test_sessions_chat_adapter.py`` pin the
event translator's behavior in isolation, but only an end-to-end
test that drives a real LLM exercises:

* :class:`SessionsChat` opening an SSE subscription against the
  in-process Omnigent server,
* :class:`omnigent.repl._repl._server_event_to_sdk_event`
  translating server-shape events into the SDK-shape events that
  the REPL renderer consumes,
* the assistant's text reaching the terminal.

If any of these regress, the spinner spins forever and no
assistant text reaches the terminal. That failure mode is silent
on every unit and integration test that does not actually
launch the REPL subprocess and read its PTY output.

Driven under a pseudo-terminal via :mod:`pexpect` because the
REPL's TUI requires a TTY to render; without one, prompt-toolkit
gracefully shuts down before the LLM response renders.

Gating:

* Reads the profile from ``OMNIGENT_SESSIONS_DEFAULT_TEST_PROFILE``
  (default ``"oss"``). Skips if the CLI cannot resolve a bearer
  for the profile.
* Skips when running under CI (``CI=true``) unless explicitly
  opted in via ``OMNIGENT_RUN_LIVE_REPL_E2E=1``. The CI profile
  is ``test-profile`` not ``oss``, and we do not want CI invocations to
  silently pay for serving capacity on every push.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pexpect
import pytest

from tests._model_pools import resolve_model
from tests.e2e.omnigent._pexpect_harness import ensure_repl_test_theme_env

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROFILE_ENV_VAR = "OMNIGENT_SESSIONS_DEFAULT_TEST_PROFILE"
_DEFAULT_PROFILE = "oss"
_MODEL = resolve_model("databricks-claude-sonnet-4-6", key=__name__)
_MARKER = "MARKER_SESSIONS_DEFAULT_42"
_REPL_TIMEOUT_S = 180
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences before substring search.

    The REPL paints via prompt-toolkit which wraps every chunk in
    SGR escapes; a raw substring match against ``MARKER`` may miss
    when the marker is broken across multiple SGR-wrapped chunks.
    """
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
    if not host:
        return None
    return host.rstrip("/")


@pytest.fixture(scope="module")
def oss_credentials() -> dict[str, str]:
    """Resolve credentials for the configured Databricks profile."""
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
            f"run `databricks auth login --profile {profile} "
            f"--host https://<workspace>.cloud.databricks.com` first, "
            f"or override the profile via {_PROFILE_ENV_VAR}=<name>.",
        )
    host = _resolve_workspace_host(profile)
    if host is None:
        pytest.skip(
            f"Databricks profile {profile!r} is missing a 'host' entry "
            f"in ~/.databrickscfg; cannot route the OpenAI client.",
        )
    return {"profile": profile, "token": token, "host": host}


def _spawn_repl(yaml_path: Path, profile: str, token: str, host: str) -> pexpect.spawn:
    """Spawn ``omnigent run`` under a PTY.

    The Databricks profile rides on the global config's ``auth:`` block
    in an isolated ``OMNIGENT_CONFIG_HOME`` plus
    ``DATABRICKS_CONFIG_PROFILE`` — the omnigent CLI no longer accepts
    ``--profile``.
    """
    config_home = Path(tempfile.mkdtemp(prefix="omnigent-sessions-default-config-"))
    (config_home / "config.yaml").write_text(
        f"auth:\n  type: databricks\n  profile: {profile}\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "DATABRICKS_TOKEN": token,
        "OPENAI_API_KEY": token,
        "OPENAI_BASE_URL": f"{host}/serving-endpoints",
        "DATABRICKS_CONFIG_PROFILE": profile,
        "OMNIGENT_CONFIG_HOME": str(config_home),
        "TERM": "xterm-256color",
        "LINES": "40",
        "COLUMNS": "120",
    }
    for k in ("ANTHROPIC_API_KEY", "CLAUDE_CODE", "CODEX"):
        env.pop(k, None)
    env = ensure_repl_test_theme_env(env)
    return pexpect.spawn(
        sys.executable,
        [
            "-m",
            "omnigent",
            "run",
            str(yaml_path),
            "--model",
            _MODEL,
        ],
        env=env,
        cwd=str(_REPO_ROOT),
        encoding="utf-8",
        timeout=_REPL_TIMEOUT_S,
        dimensions=(40, 120),
    )


def test_repl_default_sessions_renders_assistant_text(
    oss_credentials: dict[str, str],
    tmp_path: Path,
) -> None:
    """``omnigent run`` renders assistant text through sessions.

    Drives the REPL through a PTY: types the prompt, waits for the
    marker to appear on stdout, then exits via Ctrl+D. The marker
    is forced into the agent's reply via a system-prompt override
    so a positive match is unambiguous.

    Failure modes this test catches:

    * ``ResponseCreated`` (or any other SDK event class) is not
      reachable from the module path the REPL imports. The REPL
      crashes on first turn with ``ImportError``.
    * The adapter drops server-shape text events. Spinner spins,
      LLM call completes server-side, no text reaches the terminal.
    * The translator constructs an SDK envelope with the wrong
      ``response`` source class. First terminal event raises and
      aborts the stream.
    """
    yaml_path = tmp_path / "hello_world_marker.yaml"
    yaml_path.write_text(
        f"name: hello_world_marker\n"
        f"prompt: |\n"
        f"  You MUST reply with exactly the literal string\n"
        f"  {_MARKER}\n"
        f"  and nothing else. No greetings, no punctuation,\n"
        f"  no whitespace before or after the marker.\n",
    )

    child = _spawn_repl(
        yaml_path=yaml_path,
        profile=oss_credentials["profile"],
        token=oss_credentials["token"],
        host=oss_credentials["host"],
    )
    try:
        # Wait for the prompt-toolkit input affordance. ``❯`` appears
        # after banner + executor init; tolerate a long warm-up.
        child.expect("❯", timeout=60)
        child.sendline("Say the marker.")
        # Capture output until the marker appears (or timeout). The
        # marker is ASCII-only and not used by the TUI chrome, so a
        # raw substring match on PTY output is reliable. The expect
        # call returns 0 on a match; on timeout, pexpect raises.
        try:
            child.expect(_MARKER, timeout=_REPL_TIMEOUT_S)
        except pexpect.TIMEOUT:
            # On miss, dump the buffer so the failure is diagnosable.
            buf = _strip_ansi((child.before or "") + (child.after or ""))
            pytest.fail(
                f"Marker {_MARKER!r} never appeared. The REPL likely "
                f"crashed during stream rendering or the server-shape "
                f"-> SDK-shape event translator dropped the text "
                f"deltas.\n\nPTY buffer (last 4 KB, ANSI stripped):\n"
                f"{buf[-4096:]}"
            )
    finally:
        # Clean exit. Ctrl+D triggers the REPL's goodbye banner.
        try:
            child.sendcontrol("d")
            child.expect(pexpect.EOF, timeout=10)
        except pexpect.ExceptionPexpect:
            pass
        child.close(force=True)
