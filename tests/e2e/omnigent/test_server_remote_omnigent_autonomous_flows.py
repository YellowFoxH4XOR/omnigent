"""
End-to-end coverage for autonomous server-side tool flows through
``omnigent server --agent ...`` plus a remote Omnigent REPL client.

This file targets the regression where manual server mode used an
in-process runner via ``httpx.ASGITransport``. Server-side autonomous
flows such as ``sys_timer_set`` and terminal ``notify_when_idle`` can
then re-enter the Omnigent server/runner/harness stack while DBOS workflows
are active, causing the tool-result PATCH path to stall. The fix makes
``omnigent server`` start an out-of-process runner that registers
back through the WebSocket tunnel before the first turn. These tests
drive the same split topology a user runs:

  Terminal 1: ``omnigent server --agent <yaml> --port <p>``
  Terminal 2: ``omnigent run --server http://127.0.0.1:<p>``

Both tests are intentionally PTY-backed. The hang was observed in the
interactive REPL path, so a pure HTTP test would miss the runner/REPL
wiring that matters here.
"""

from __future__ import annotations

import contextlib
import io
import os
import secrets
import shutil
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import httpx
import pexpect
import pytest

from omnigent.entities.conversation import MessageData
from omnigent.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from tests._model_pools import resolve_model
from tests.e2e.omnigent._pexpect_harness import (
    ensure_repl_test_theme_env,
    strip_ansi,
    submit_prompt,
)

_MODEL = resolve_model("databricks-gpt-5-mini", key=__name__)
_HARNESS = "openai-agents"
_TERM = "xterm-256color"
_ROWS = 40
_COLS = 120

_SERVER_BOOT_TIMEOUT = 60.0
_REPL_BOOT_TIMEOUT = 90.0
_TURN_TIMEOUT = 180.0
_AUTONOMOUS_TIMEOUT = 180.0

_TIMER_PROMPT = "Set a 5 seconds timer with note remote-e2e."
_IDLE_PROMPT = "start"

_TIMER_YAML = """\
name: remote_timer_e2e
prompt: |
  You are testing sys_timer_set in a remote Omnigent client connected to a
  manually started omnigent server.

  When asked to set a timer, call sys_timer_set exactly once with the
  requested seconds and note. After the tool returns, reply with the
  literal text TIMER_SCHEDULED. Do not wait for the timer to fire.

timers: true

executor:
  model: databricks-gpt-5-mini
  harness: openai-agents
"""

_IDLE_YAML = """\
name: remote_idle_e2e
prompt: |
  You are testing terminal idle notifications in a remote Omnigent client
  connected to a manually started omnigent server. Do EXACTLY these
  steps when the user says start:

  1. Call sys_terminal_launch(terminal="demo", session="s1",
     notify_when_idle=true). Verify the response includes
     "notify_when_idle": true.
  2. Call sys_terminal_send(terminal="demo", session="s1",
     text="sleep 30 && echo done", keys="Enter").
  3. Call sys_os_shell(command="sleep 18"). Do not skip this step.
     While you are blocked here, the terminal watcher should observe
     the demo terminal as idle and deliver an idle notification.
  4. Call sys_terminal_close(terminal="demo", session="s1").
  5. Reply with the literal text IDLE_DONE.

executor:
  model: databricks-gpt-5-mini
  harness: openai-agents

os_env:
  type: caller_process
  cwd: .
  sandbox:
    type: none

terminals:
  demo:
    command: bash
"""


@contextlib.contextmanager
def _short_tmpdir(prefix: str) -> Iterator[Path]:
    """
    Create a short path under ``/tmp``.

    Terminal tests need a short TMPDIR because tmux Unix socket paths
    include nested random directories. macOS caps Unix socket paths at
    roughly 104 chars, so pytest's deeply nested ``tmp_path`` can make
    ``sys_terminal_launch`` fail before it reaches the behavior under
    test.
    """
    path = Path("/tmp") / f"{prefix}-{uuid.uuid4().hex[:6]}"
    path.mkdir()
    try:
        yield path
    except BaseException:
        print(f"\n[remote-autonomous debug] tmpdir preserved at {path}")
        raise
    else:
        shutil.rmtree(path, ignore_errors=True)


def _find_free_port() -> int:
    """Return a currently-free loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextlib.contextmanager
def _manual_server(
    *,
    omnigent_python: Path,
    omnigent_repo_root: Path,
    yaml_path: Path,
    port: int,
    db_path: Path,
    artifact_dir: Path,
    env: dict[str, str],
    runner_id: str,
    binding_token: str,
) -> Iterator[subprocess.Popen[str]]:
    """
    Spawn the real manual server command with one pre-registered agent
    and a sibling runner subprocess.

    Uses ``python -m omnigent.cli server`` rather than importing server
    internals so the test covers the CLI path that routes server-to-runner
    calls over the WS tunnel.
    """
    base_url = f"http://127.0.0.1:{port}"
    log_path = db_path.parent / "server.log"
    log_fh = log_path.open("w")
    proc = subprocess.Popen(
        [
            str(omnigent_python),
            "-m",
            "omnigent.cli",
            "server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--database-uri",
            f"sqlite:///{db_path}",
            "--artifact-location",
            str(artifact_dir),
            "--agent",
            str(yaml_path),
        ],
        env={**env, "OMNIGENT_RUNNER_TUNNEL_TOKEN": binding_token},
        cwd=str(omnigent_repo_root),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_fh.close()

    # Spawn runner as sibling subprocess.
    runner_log_path = db_path.parent / "runner.log"
    runner_log_fh = runner_log_path.open("w")
    runner_proc = subprocess.Popen(
        [str(omnigent_python), "-m", "omnigent.runner._entry"],
        env={
            **env,
            "OMNIGENT_RUNNER_ID": runner_id,
            "OMNIGENT_RUNNER_TUNNEL_BINDING_TOKEN": binding_token,
            "OMNIGENT_RUNNER_PARENT_PID": str(os.getpid()),
            "RUNNER_SERVER_URL": base_url,
        },
        cwd=str(omnigent_repo_root),
        stdout=runner_log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    runner_log_fh.close()

    try:
        _wait_for_server(
            port,
            proc=proc,
            log_path=log_path,
            runner_id=runner_id,
        )
        yield proc
    finally:
        if runner_proc.poll() is None:
            runner_proc.send_signal(signal.SIGTERM)
            try:
                runner_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                runner_proc.kill()
                runner_proc.wait(timeout=5)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def _wait_for_server(
    port: int,
    *,
    proc: subprocess.Popen[str],
    log_path: Path,
    runner_id: str | None,
) -> None:
    """Poll server health and runner status until the manual server is ready.

    :param port: Loopback port used by the manual server.
    :param proc: Manual server subprocess.
    :param log_path: Captured server log path.
    :param runner_id: Expected runner id, e.g.
        ``"runner_remote_timer_test"``. When provided, readiness
        requires ``/v1/runners/{runner_id}/status`` to report online.
    :returns: None.
    """
    deadline = time.monotonic() + _SERVER_BOOT_TIMEOUT
    last_error = "not polled yet"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = log_path.read_text() if log_path.exists() else "<missing server.log>"
            raise AssertionError(
                f"manual omnigent server exited early with code {proc.returncode}.\n"
                f"Server log:\n{output[-6000:]}"
            )
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                if runner_id is None:
                    return
                try:
                    runner_resp = httpx.get(
                        f"http://127.0.0.1:{port}/v1/runners/{runner_id}/status",
                        timeout=2.0,
                    )
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException) as exc:
                    last_error = f"runner status {type(exc).__name__}: {exc}"
                else:
                    if runner_resp.status_code == 200 and runner_resp.json()["online"] is True:
                        return
                    last_error = (
                        f"runner status HTTP {runner_resp.status_code}: {runner_resp.text[:200]}"
                    )
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        time.sleep(0.2)
    output = log_path.read_text() if log_path.exists() else "<missing server.log>"
    pytest.fail(
        f"manual omnigent server did not become ready within "
        f"{_SERVER_BOOT_TIMEOUT}s (last_error={last_error}).\n"
        f"Server log:\n{output[-6000:]}"
    )


def _spawn_remote_repl(
    *,
    omnigent_python: Path,
    omnigent_repo_root: Path,
    port: int,
    env: dict[str, str],
) -> pexpect.spawn:
    """Spawn ``omnigent run --server http://...`` under a PTY."""
    args = [
        "-m",
        "omnigent.cli",
        "run",
        "--server",
        f"http://127.0.0.1:{port}",
    ]
    spawn_env = ensure_repl_test_theme_env(env)
    return pexpect.spawn(
        str(omnigent_python),
        args,
        env={
            **spawn_env,
            "TERM": _TERM,
            "LINES": str(_ROWS),
            "COLUMNS": str(_COLS),
        },
        cwd=str(omnigent_repo_root),
        encoding="utf-8",
        timeout=60.0,
        dimensions=(_ROWS, _COLS),
    )


def _run_remote_prompt(
    *,
    omnigent_python: Path,
    omnigent_repo_root: Path,
    port: int,
    env: dict[str, str],
    prompt: str,
    done_text: str,
    turn_timeout: float = _TURN_TIMEOUT,
) -> str:
    """
    Open a remote Omnigent REPL, submit one prompt, wait for ``done_text``, exit.

    Returns the ANSI-stripped captured PTY output for failure assertions.
    """
    child = _spawn_remote_repl(
        omnigent_python=omnigent_python,
        omnigent_repo_root=omnigent_repo_root,
        port=port,
        env=env,
    )
    captured = io.StringIO()
    child.logfile_read = captured
    try:
        child.expect(r"❯ ", timeout=_REPL_BOOT_TIMEOUT)
        submit_prompt(child, prompt)
        child.expect(done_text, timeout=turn_timeout)
        # Force-close after done_text is matched. The timer / idle tests
        # may have autonomous server-side activity in flight (e.g. a timer
        # callback starting a new LLM turn) when done_text appears, so a
        # graceful Ctrl+D exit would time out waiting for those turns to
        # complete. We've already captured the key observable; close now.
    finally:
        if not child.closed:
            child.close(force=True)
    return strip_ansi(captured.getvalue())


def _conversation_texts(db_path: Path) -> list[str]:
    """Return all message text blocks from the server conversation DB."""
    conv_store = SqlAlchemyConversationStore(f"sqlite:///{db_path}")
    texts: list[str] = []
    convs = conv_store.list_conversations(limit=50)
    for conv in convs.data:
        page = conv_store.list_items(conversation_id=conv.id, limit=500)
        for item in page.data:
            if item.type != "message" or not isinstance(item.data, MessageData):
                continue
            for block in item.data.content or []:
                if not isinstance(block, dict):
                    continue
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
    return texts


def _wait_for_conversation_text(
    *,
    db_path: Path,
    needle: str,
    timeout: float,
    server_log_path: Path,
    repl_output: str,
) -> None:
    """Poll the conversation DB until a message containing ``needle`` appears."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(needle in text for text in _conversation_texts(db_path)):
            return
        time.sleep(0.5)

    server_log_tail = ""
    if server_log_path.exists():
        server_log_tail = server_log_path.read_text()[-6000:]
    texts = _conversation_texts(db_path)
    pytest.fail(
        f"Expected to find {needle!r} in conversation messages at {db_path}, "
        f"but it did not appear within {timeout}s.\n\n"
        f"All message texts:\n{texts!r}\n\n"
        f"REPL output tail:\n{repl_output[-4000:]}\n\n"
        f"Server log tail:\n{server_log_tail}"
    )


def test_manual_server_remote_omnigent_sys_timer_set_does_not_hang_and_fires(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    databricks_workspace: tuple[str, str],
) -> None:
    """
    A remote Omnigent REPL connected to ``omnigent server`` can schedule a timer.

    Regression signal: before the runner-topology fix, the first turn could
    wedge while trying to return the ``sys_timer_set`` result through the
    in-process runner. This test fails fast if ``TIMER_SCHEDULED`` never
    appears, and then additionally verifies the background timer firing row
    lands in the server's conversation DB.
    """
    from omnigent.runner.identity import token_bound_runner_id

    with _short_tmpdir("oa-remote-timer") as workdir:
        yaml_path = workdir / "timer.yaml"
        yaml_path.write_text(_TIMER_YAML)
        db_path = workdir / "chat.db"
        artifact_dir = workdir / "artifacts"
        port = _find_free_port()
        binding_token = secrets.token_urlsafe(32)
        runner_id = token_bound_runner_id(binding_token)
        env = {
            **omnigent_credentials_env,
            "TMPDIR": str(workdir),
        }

        with _manual_server(
            omnigent_python=omnigent_python,
            omnigent_repo_root=omnigent_repo_root,
            yaml_path=yaml_path,
            port=port,
            db_path=db_path,
            artifact_dir=artifact_dir,
            env=env,
            runner_id=runner_id,
            binding_token=binding_token,
        ):
            repl_output = _run_remote_prompt(
                omnigent_python=omnigent_python,
                omnigent_repo_root=omnigent_repo_root,
                port=port,
                env=env,
                prompt=_TIMER_PROMPT,
                done_text="TIMER_SCHEDULED",
            )
            assert "sys_timer_set" in repl_output, (
                "Expected the REPL transcript to show sys_timer_set was called.\n"
                f"Output tail:\n{repl_output[-4000:]}"
            )
            _wait_for_conversation_text(
                db_path=db_path,
                needle="remote-e2e",
                timeout=_AUTONOMOUS_TIMEOUT,
                server_log_path=workdir / "server.log",
                repl_output=repl_output,
            )


def test_manual_server_remote_omnigent_notify_when_idle_does_not_hang_and_delivers(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
    databricks_workspace: tuple[str, str],
) -> None:
    """
    A remote Omnigent REPL connected to ``omnigent server`` can use
    ``sys_terminal_launch(..., notify_when_idle=true)``.

    The prompt keeps the parent turn alive while the terminal goes quiet, so
    the idle watcher should deliver ``[System: terminal demo:s1 is idle]`` to
    the conversation. The main regression signal is that the turn reaches
    ``IDLE_DONE`` instead of wedging in the runner/harness path.
    """
    from omnigent.runner.identity import token_bound_runner_id

    with _short_tmpdir("oa-remote-idle") as workdir:
        yaml_path = workdir / "idle.yaml"
        yaml_path.write_text(_IDLE_YAML)
        db_path = workdir / "chat.db"
        artifact_dir = workdir / "artifacts"
        port = _find_free_port()
        binding_token = secrets.token_urlsafe(32)
        runner_id = token_bound_runner_id(binding_token)
        env = {
            **omnigent_credentials_env,
            "TMPDIR": str(workdir),
        }

        with _manual_server(
            omnigent_python=omnigent_python,
            omnigent_repo_root=omnigent_repo_root,
            yaml_path=yaml_path,
            port=port,
            db_path=db_path,
            artifact_dir=artifact_dir,
            env=env,
            runner_id=runner_id,
            binding_token=binding_token,
        ):
            repl_output = _run_remote_prompt(
                omnigent_python=omnigent_python,
                omnigent_repo_root=omnigent_repo_root,
                port=port,
                env=env,
                prompt=_IDLE_PROMPT,
                done_text="IDLE_DONE",
                turn_timeout=240.0,
            )
            assert "sys_terminal_launch" in repl_output, (
                "Expected the REPL transcript to show sys_terminal_launch was called.\n"
                f"Output tail:\n{repl_output[-4000:]}"
            )
            _wait_for_conversation_text(
                db_path=db_path,
                needle="[System: terminal demo:s1 is idle]",
                timeout=30.0,
                server_log_path=workdir / "server.log",
                repl_output=repl_output,
            )
