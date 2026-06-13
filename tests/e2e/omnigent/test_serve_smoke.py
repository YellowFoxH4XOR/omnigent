"""Phase 0 characterization test — ``omnigent server --agent <agent>`` smoke.

Boots ``omnigent server --agent <yaml>`` as a real subprocess, polls
the agent-info endpoint until the server responds, sends one
``POST /api/chat``, asserts the response shape matches the
captured snapshot, and kills the server. One test per
user-facing surface; the per-flag matrix (``--model``,
``--harness``, ``--profile``, etc.) lands in follow-ups.

**What breaks if this fails:**
- ``omnigent server --agent <agent>`` fails to boot the Starlette app
  (``omnigent.server.create_app``).
- The ``/api/agent`` handler regresses its response schema
  (``name`` field disappears, ``mascot`` layout changes, etc.).
- The ``/api/chat`` handler's JSON contract regresses (missing
  ``response`` / ``session_id`` fields, 500 on a trivial
  prompt).
- ``RequestHandler.handle_chat`` loses its ability to route to
  the ``openai-agents`` harness end-to-end.
- The server fails to honor ``--port`` and binds to the default
  8000 instead (a test on a dev box would then race with
  whatever is already on 8000).

Design reference: ``designs/OMNIGENT_INTEGRATION.md`` §Phase 0
``omnigent server --agent <agent>`` HTTP characterization.
"""

from __future__ import annotations

import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests._model_pools import resolve_model
from tests.e2e.omnigent._snapshot import compare_snapshot

# Model + harness chosen so no ``~/.databrickscfg`` patching is
# required — ``openai-agents`` honors ``OPENAI_BASE_URL`` /
# ``OPENAI_API_KEY`` directly.
_MODEL = resolve_model("databricks-gpt-5-mini", key=__name__)
_HARNESS = "openai-agents"

# ``hello_world.yaml`` is the minimum agent that exercises the
# serve pipeline (loads YAML → creates executor → handles one
# chat call). Tool-bearing YAMLs would add orthogonal variance
# the smoke test shouldn't own.
_YAML_RELPATH = ("examples", "hello_world.yaml")

# Seconds allocated to the server to open its listening socket
# and respond on ``/api/agent``. Omnigent' serve path imports
# uvicorn + the harness SDK on startup; on cold caches this can
# take several seconds.
_SERVE_BOOT_TIMEOUT = 30.0

# Seconds allocated to the chat call. One LLM round-trip — 60s
# is ample for ``databricks-gpt-5-mini`` on a "hi" prompt.
_CHAT_TIMEOUT = 60.0


def find_free_port() -> int:
    """
    Return a free TCP port on localhost.

    Equivalent to ``tests/e2e/conftest.py``'s helper of the same
    name; duplicated here rather than imported because
    ``test_serve_smoke`` is the only member of the
    ``omnigent`` suite that needs it and the duplication is
    tiny.

    :returns: A port number that was free at the instant this
        function returned, e.g. ``58123``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _omnigent_serve(
    *,
    omnigent_python: Path,
    yaml_path: Path,
    port: int,
    env: dict[str, str],
    cwd: Path,
) -> Iterator[subprocess.Popen[str]]:
    """
    Spawn ``omnigent server --agent <agent>`` as a subprocess and tear it down
    cleanly on exit.

    :param omnigent_python: Interpreter that has omnigent
        installed.
    :param yaml_path: Absolute path to the agent YAML.
    :param port: Bind port.
    :param env: Subprocess environment (PAT + base URL already
        populated).
    :param cwd: Working directory for the subprocess.
    :yields: The live subprocess. Caller holds it while making
        HTTP calls. Teardown sends SIGTERM, then SIGKILL after
        10s if the process is still running.
    """
    proc = subprocess.Popen(
        [
            str(omnigent_python),
            "-m",
            "omnigent",
            "serve",
            str(yaml_path),
            "--port",
            str(port),
            "--model",
            _MODEL,
            "--harness",
            _HARNESS,
            # Disable the chat web UI so we're characterizing
            # the REST surface only — the web UI is orthogonal
            # to the integration and out of scope for phase 0.
            "--no-web",
        ],
        env=env,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        yield proc
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_serve_smoke_one_chat_round_trip(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    ``omnigent server --agent <agent>`` boots, responds on ``/api/agent``, and
    answers one ``/api/chat`` POST with the expected JSON shape.

    :param omnigent_python: Interpreter with omnigent +
        openai-agents installed.
    :param omnigent_repo_root: Working directory for the
        subprocess — needed so the YAML loader resolves
        ``callable:`` dotted-path tools (for future YAMLs;
        hello_world has none).
    :param omnigent_credentials_env: Env vars populated with
        the PAT and base URL.
    """
    port = find_free_port()
    yaml_path = omnigent_repo_root.joinpath(*_YAML_RELPATH)
    with _omnigent_serve(
        omnigent_python=omnigent_python,
        yaml_path=yaml_path,
        port=port,
        env=omnigent_credentials_env,
        cwd=omnigent_repo_root,
    ) as proc:
        _wait_for_agent_info(port, timeout=_SERVE_BOOT_TIMEOUT, proc=proc)
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            timeout=_CHAT_TIMEOUT,
        ) as client:
            info_resp = client.get("/api/agent")
            info_resp.raise_for_status()
            info_body = info_resp.json()
            chat_resp = client.post("/api/chat", json={"message": "say hi in 5 words"})
            chat_resp.raise_for_status()
            chat_body = chat_resp.json()

    # ``info_body`` / ``chat_body`` come from a fresh
    # ``omnigent server --agent <agent>`` that we just proved is live. Any key
    # missing here indicates a contract regression — use
    # direct [] access so the KeyError surfaces loud instead of
    # being masked by a default.
    observed: dict[str, Any] = {
        "agent_info_status": info_resp.status_code,
        "agent_info_name": info_body["name"],
        "agent_info_has_mascot": "mascot" in info_body,
        "chat_status": chat_resp.status_code,
        "chat_has_response": isinstance(chat_body.get("response"), str),
        "chat_has_session_id": isinstance(chat_body.get("session_id"), str),
        "chat_response_text": chat_body["response"],
    }
    diffs = compare_snapshot("test_serve_smoke", observed)
    assert diffs == [], (
        "Snapshot mismatch for omnigent serve smoke:\n"
        + "\n".join(diffs)
        + f"\n\nagent_info body: {info_body!r}\nchat body: {chat_body!r}"
    )


def _wait_for_agent_info(port: int, *, timeout: float, proc: subprocess.Popen[str]) -> None:
    """
    Poll ``GET /api/agent`` until the server responds 200.

    Omnigent' serve path doesn't expose a dedicated
    ``/health`` endpoint, so we use ``/api/agent`` as a readiness
    probe — it requires the server, the session manager, and
    the executor to have finished initializing. Same polling
    pattern as the parent conftest's ``wait_for_server``.

    :param port: The bound port passed via ``--port``.
    :param timeout: Max seconds to wait before failing the
        test.
    :param proc: The server subprocess; inspected on every
        poll iteration. If it has exited before we see ready,
        that's a hard error with its captured stdout surfaced.
    :raises AssertionError: If the server exits early.
    :raises pytest.fail.Exception: If the server doesn't
        respond within ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout is not None else "<no output>"
            raise AssertionError(
                f"omnigent serve exited early with code "
                f"{proc.returncode} before /api/agent became ready.\n\n"
                f"Server output:\n{output}"
            )
        # Only catch httpx transport errors: the server isn't
        # listening yet / the connection races a teardown. Any
        # OTHER exception (schema mismatch, SSL misconfig) is a
        # real bug and should surface, not be swallowed.
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/api/agent", timeout=2.0)
        except (httpx.ConnectError, httpx.ReadError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                return
            last_error = f"HTTP {resp.status_code}"
        # Event-loop-friendly poll interval. Mirrors the parent
        # conftest's ``wait_for_server`` pattern; no other way
        # to wait on a subprocess's port-binding without
        # restructuring omnigent' serve entrypoint.
        # Unavoidable — readiness probe for a subprocess;
        # restructuring omnigent.serve to expose an "async"
        # boot signal is outside phase 0's scope.
        time.sleep(0.3)
    pytest.fail(
        f"omnigent serve did not respond on /api/agent within "
        f"{timeout}s (last_error={last_error!r})."
    )
