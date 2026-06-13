"""Phase 3 integration-code test — ``omnigent server --agent`` actually
routes to an omnigent server, not the legacy omnigent server.

This file hits an endpoint unique to Omnigent' surface
(``/v1/sessions`` list) and asserts on an omnigent-shaped error
from ``/v1/responses`` that the legacy omnigent server cannot
produce.

**What breaks if this fails:**
- The Omnigent mode dispatch site at ``_serve_agent`` stops calling into
  omnigent and falls back to the legacy ``create_app`` (the
  "silent fallback" antipattern the design forbids).
- The shim's ``_omnigent_register_yaml_bundle`` stops actually registering
  the synthesized bundle with Omnigent' ``AgentStore`` (server
  boots but ``/v1/sessions`` returns empty).
- The shim's YAML translation pipeline regresses — e.g. the
  defaulted harness field drops out and Omnigent' validator
  rejects the bundle at registration time, so the server starts
  but no agent shows up.
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

from tests.e2e.omnigent._snapshot import compare_snapshot

# hello_world.yaml is the minimum YAML that exercises the shim —
# no tools, no policies, no executor block (so the shim's "default
# harness when YAML omits one" path is also covered).
_YAML_RELPATH = ("tests", "resources", "examples", "hello_world.yaml")

# Cold-start budget for the omnigent server. Needs headroom for
# DBOS sqlite migrations + uvicorn startup + shim's bundle
# registration round-trip — 30s matches the sibling test_serve_*
# files.
_SERVE_BOOT_TIMEOUT = 30.0

# httpx timeout for the individual probe round-trips the test does.
_HTTP_TIMEOUT = 10.0

# Readiness-poll sleep. Short enough that a cold-start subprocess
# does not wait past the deadline due to sparse polling, long
# enough that we do not burn CPU.
_POLL_INTERVAL_S = 0.3


@contextmanager
def _omnigent_serve_omnigent(
    *,
    omnigent_python: Path,
    yaml_path: Path,
    port: int,
    env: dict[str, str],
    cwd: Path,
) -> Iterator[subprocess.Popen[str]]:
    """
    Spawn ``omnigent server --agent <yaml> --port <port>`` as a
    subprocess and tear it down on context-manager exit.

    :param omnigent_python: Interpreter with ``omnigent``
        installed, from :mod:`tests.e2e.omnigent.conftest`.
    :param yaml_path: Absolute path to the YAML to serve, e.g.
        ``Path(".../examples/hello_world.yaml")``.
    :param port: Bind port passed as ``--port``.
    :param env: Subprocess environment (PAT + base URL already
        populated by the fixture).
    :param cwd: Working directory for the subprocess.
    :yields: The live subprocess. Teardown sends SIGTERM then
        SIGKILL at 10s.
    """
    proc = subprocess.Popen(
        [
            str(omnigent_python),
            "-m",
            "omnigent",
            "server",
            "--agent",
            str(yaml_path),
            "--port",
            str(port),
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


def _wait_for_health(
    port: int,
    *,
    timeout: float,
    proc: subprocess.Popen[str],
) -> None:
    """
    Poll Omnigent' ``/health`` until the server responds 200.

    Omnigent exposes ``/health`` as a trivial readiness probe
    that doesn't require any agent to be registered.

    :param port: Bound port.
    :param timeout: Max seconds to wait.
    :param proc: The server subprocess; checked on every poll.
    :raises AssertionError: If the subprocess exits before ready.
    :raises pytest.fail.Exception: On timeout.
    """
    deadline = time.monotonic() + timeout
    last_error: str | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout is not None else "<no output>"
            raise AssertionError(
                f"omnigent server --agent exited early with code "
                f"{proc.returncode} before /health became ready.\n\n"
                f"Server output:\n{output}"
            )
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
        except (httpx.ConnectError, httpx.ReadError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if resp.status_code == 200:
                return
            last_error = f"HTTP {resp.status_code}"
        time.sleep(_POLL_INTERVAL_S)
    pytest.fail(
        f"omnigent server --agent did not respond on /health within "
        f"{timeout}s (last_error={last_error!r})."
    )


def _find_free_port() -> int:
    """
    Return a free TCP port on localhost.

    :returns: A port number that was free at the instant this
        function returned, e.g. ``58123``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gather_omnigent_observations(port: int) -> dict[str, Any]:
    """
    Capture structural observations that prove the booted server is
    omnigent, not legacy omnigent.

    Observations:

    - ``health_status``: ``GET /health`` returns 200. Legacy
      omnigent has no ``/health`` route — the legacy serve
      characterization proved ``/api/agent`` is the only reliable
      readiness probe. A legacy server accidentally left in place
      would 404 here.
    - ``agents_list_status``: ``GET /v1/agents`` returns 200.
      Omnigent exposes the built-in agent catalog; legacy omnigent
      does not. A 200 here proves we're talking to omnigent.
    - ``agents_has_hello_world``: ``True`` — proves the server's
      ``--agent`` pre-registration path carried the YAML's ``name``
      field through to the built-in catalog.
    - ``responses_unknown_agent_status``: ``POST /v1/responses``
      reaches Omnigent' OpenResponses surface and rejects an unknown
      agent id with 404. Legacy omnigent has no ``/v1/responses``
      endpoint at all.

    :param port: Port the subprocess is bound to.
    :returns: Snapshot-friendly observations.
    """
    with httpx.Client(
        base_url=f"http://127.0.0.1:{port}",
        timeout=_HTTP_TIMEOUT,
    ) as client:
        health_resp = client.get("/health")
        agents_resp = client.get("/v1/agents")
        # Direct [] access for body keys — KeyError surfaces a
        # contract regression loudly rather than being masked by
        # a default.
        agents_body = agents_resp.json()
        agents_data = agents_body["data"]
        agent_names = [item["name"] for item in agents_data]
        incomplete_payload = {
            "agent_id": "missing",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "stream": False,
            "background": False,
        }
        responses_resp = client.post(
            "/v1/responses",
            json=incomplete_payload,
        )
    return {
        "health_status": health_resp.status_code,
        "agents_list_status": agents_resp.status_code,
        "agents_has_hello_world": "hello_world" in agent_names,
        "responses_unknown_agent_status": responses_resp.status_code,
    }


def test_serve_omnigent_routes_to_omnigent(
    omnigent_python: Path,
    omnigent_repo_root: Path,
    omnigent_credentials_env: dict[str, str],
) -> None:
    """
    ``omnigent server --agent <yaml>`` boots an omnigent server
    with the YAML pre-registered.

    The assertion ensemble proves both (a) it's Omnigent'
    server (by surfacing an endpoint and error shape legacy
    omnigent can't produce) and (b) the shim's registration
    pipeline succeeded (the YAML's ``name`` surfaces through
    ``/v1/agents``).

    :param omnigent_python: Interpreter with omnigent and
        omnigent installed.
    :param omnigent_repo_root: Cwd for the subprocess.
    :param omnigent_credentials_env: PAT + base URL env. The
        Omnigent path doesn't make LLM calls in this test, but the
        server lifecycle still touches the sqlite DB + artifact
        dir; passing the standard env keeps the test mechanically
        parallel to the rest of the suite.
    """
    port = _find_free_port()
    yaml_path = omnigent_repo_root.joinpath(*_YAML_RELPATH)
    with _omnigent_serve_omnigent(
        omnigent_python=omnigent_python,
        yaml_path=yaml_path,
        port=port,
        env=omnigent_credentials_env,
        cwd=omnigent_repo_root,
    ) as proc:
        _wait_for_health(port, timeout=_SERVE_BOOT_TIMEOUT, proc=proc)
        observed = _gather_omnigent_observations(port)

    diffs = compare_snapshot("test_serve_omnigent_routes", observed)
    assert diffs == [], (
        "Snapshot mismatch for omnigent server --agent routing:\n"
        + "\n".join(diffs)
        + f"\n\nObserved: {observed!r}"
    )
