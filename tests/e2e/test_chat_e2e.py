"""E2E test for ``omnigent chat`` — local mode with archer.

Verifies that ``omnigent chat ./agent-dir/`` starts a server, opens the
REPL, and the agent responds. Since the REPL is interactive, we
test by directly calling the local mode components rather than
launching the full CLI.

Usage::

    pytest tests/e2e/test_chat_e2e.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.e2e.conftest import find_free_port, wait_for_server


def _resolve_workspace(request: pytest.FixtureRequest) -> tuple[str, str]:
    """
    Resolve ``(profile, host)`` from the active ``--profile``.

    Falls back to ``default`` (the profile CI passes explicitly)
    when ``--profile`` isn't given; raises if the profile isn't in
    ``~/.databrickscfg``. Local helper rather than a shared
    fixture so this file stays self-contained.

    :param request: pytest request — used to read ``--profile``.
    :returns: ``(profile_name, host_url)``. Host has trailing
        ``/`` stripped.
    :raises pytest.UsageError: When the resolved profile isn't
        configured.
    """
    profile = request.config.getoption("--profile") or "default"
    cfg_path = Path.home() / ".databrickscfg"
    cfg = configparser.ConfigParser()
    if cfg_path.exists():
        cfg.read(cfg_path)
    if profile not in cfg or not cfg[profile].get("host"):
        raise pytest.UsageError(
            f"Databricks profile {profile!r} is missing from {cfg_path} or has no ``host`` entry."
        )
    return profile, cfg[profile]["host"].rstrip("/")


# Path was ``examples/agents/archer/`` before the layout
# flattening (commit 3abd7c2 "examples: flatten examples/agents/
# → examples/"); now ``examples/archer/``.
_ARCHER_DIR = Path(__file__).resolve().parents[2] / "examples" / "archer"


# ``body`` is a parsed Responses API JSON object with heterogeneous
# nested shape (output items, content parts, etc.); using ``Any``
# here avoids drawing up a TypedDict for a test helper that just
# walks the tree looking for output_text values.
def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all output_text blocks from a response body.

    :param body: The terminal response body.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)


def test_chat_local_starts_server_and_agent_responds(
    llm_api_key: str,
    openai_judge_api_key: str,
) -> None:
    """
    ``omnigent chat ./agent-dir/`` starts a local server with the agent
    and the agent can respond to messages.

    Tests the server startup and agent registration path used by
    ``omnigent chat`` in local mode. Since the REPL itself is interactive,
    we verify the underlying server works by sending a direct HTTP
    request.

    **What breaks if this fails:**
    - _start_local_server broken → server doesn't boot.
    - Agent bundle not registered → 404 on responses.
    - Agent config invalid → 500 on responses.
    """
    from omnigent.chat import (
        _start_local_server,
        _stop_local_server,
        _wait_for_server,
    )

    # The archer spec's connection block uses ${OPENAI_API_KEY}, which
    # the spec parser expands at load time from the subprocess's env.
    os.environ["OPENAI_API_KEY"] = openai_judge_api_key

    port = find_free_port()
    server = _start_local_server(_ARCHER_DIR, port)

    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_server(port, server)

        # Verify agent is registered by checking for sessions with
        # the expected agent_name.
        sessions_resp = httpx.get(
            f"{base_url}/v1/sessions",
            params={"agent_name": "archer", "limit": 1},
            timeout=10.0,
        )
        sessions_resp.raise_for_status()
        sessions = sessions_resp.json()["data"]
        assert len(sessions) > 0, "No sessions with agent_name='archer' after server start."

        agent_name = sessions[0]["agent_name"]
        assert agent_name == "archer", f"Expected archer agent, got {agent_name!r}."

        # Send a message and verify the agent responds.
        resp = httpx.post(
            f"{base_url}/v1/responses",
            json={
                "model": agent_name,
                "input": "Say hello briefly.",
                "stream": False,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        body = resp.json()

        assert body["status"] == "completed", (
            f"Status: {body['status']!r}. Output: {body.get('output', [])}"
        )

        text = _extract_all_text(body)
        # Verify the agent actually produced non-whitespace text, not
        # just an empty or whitespace-only response that len() > 0
        # would let through.
        assert text.strip(), f"Agent produced no meaningful text output. Raw: {text!r}"

    finally:
        _stop_local_server(server)


def test_chat_local_accepts_omnigent_yaml_file(
    llm_api_key: str,
    openai_judge_api_key: str,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """
    ``omnigent chat examples/coding_supervisor.yaml`` (or any
    standalone omnigent YAML) now starts the local server and
    registers the agent under its spec-declared name.

    The YAML path exercises the new ``materialize_bundle`` code
    path in :func:`_preregister_agent`: a file source wraps into
    a bundle directory, gets tarred, and the stored tarball
    round-trips through :func:`omnigent.spec.load` to a
    validated :class:`AgentSpec`.

    **What breaks if this fails:**
    - ``_preregister_agent`` regresses to directory-only.
    - ``materialize_bundle``'s file branch produces the wrong
      dir shape and ``_find_omnigent_yaml_in_dir`` misses the
      YAML.
    - Agent-plane's spec dispatch stops routing omnigent YAMLs
      through ``load_omnigent_yaml``.

    :param llm_api_key: Databricks PAT passed as
        ``OPENAI_API_KEY`` to the openai-agents harness.
    :param tmp_path: Per-test temp dir for the YAML fixture.
    """
    import yaml as _yaml

    from omnigent.chat import (
        _start_local_server,
        _stop_local_server,
        _wait_for_server,
    )

    # Inline fixture: minimal omnigent YAML with harness set so
    # the synthesized spec passes the validator. Self-contained
    # so an edit to the real ``examples/hello_world.yaml`` can't
    # flake this test.
    profile, host = _resolve_workspace(request)
    yaml_path = tmp_path / "yaml-e2e-probe.yaml"
    yaml_path.write_text(
        _yaml.safe_dump(
            {
                "name": "yaml-e2e-probe",
                "prompt": "You are a friendly assistant. Say hello briefly.",
                "executor": {
                    "model": "databricks-gpt-5-mini",
                    "harness": "openai-agents",
                    "profile": profile,
                },
            },
        ),
    )

    # openai-agents reads credentials from ``OPENAI_API_KEY`` and
    # its endpoint from ``OPENAI_BASE_URL`` when no explicit profile
    # wins. Point both at Databricks' native OpenAI Responses AI
    # Gateway so the PAT authenticates — without the base-URL
    # override the SDK hits api.openai.com and the PAT is rejected
    # as a malformed OpenAI key.
    os.environ["OPENAI_API_KEY"] = openai_judge_api_key
    os.environ["OPENAI_BASE_URL"] = f"{host}/ai-gateway/openai/v1"

    port = find_free_port()
    server = _start_local_server(yaml_path, port)
    base_url = f"http://127.0.0.1:{port}"

    try:
        _wait_for_server(port, server)

        # Agent is registered under the YAML's ``name`` field —
        # proves the materialize_bundle → tarball → load chain
        # preserves the spec name end-to-end.
        sessions_resp = httpx.get(
            f"{base_url}/v1/sessions",
            params={"agent_name": "yaml-e2e-probe", "limit": 1},
            timeout=10.0,
        )
        sessions_resp.raise_for_status()
        sessions = sessions_resp.json()["data"]
        assert len(sessions) == 1, (
            f"Expected exactly one session with agent_name='yaml-e2e-probe', got {len(sessions)}."
        )
        assert sessions[0]["agent_name"] == "yaml-e2e-probe", (
            f"Expected 'yaml-e2e-probe' (from YAML name field), got {sessions[0]['agent_name']!r}."
        )

        # A full turn proves the spec the server rehydrates from
        # the stored tarball also produces a runnable agent — not
        # just a registered-but-broken one. This is the single
        # strongest regression guard for the bundling refactor.
        resp = httpx.post(
            f"{base_url}/v1/responses",
            json={
                "model": "yaml-e2e-probe",
                "input": "Say hello briefly.",
                "stream": False,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        body = resp.json()

        assert body["status"] == "completed", (
            f"Status: {body['status']!r}. Output: {body.get('output', [])}"
        )
        assert _extract_all_text(body).strip(), (
            f"Agent produced no meaningful output. Body: {body!r}"
        )
    finally:
        _stop_local_server(server)


def test_chat_remote_pick_agent(
    llm_api_key: str,
    openai_judge_api_key: str,
) -> None:
    """
    Remote chat can list and identify agents on a server.

    Tests the remote mode's agent discovery by starting a server with
    archer and verifying ``_pick_agent`` finds it.

    **What breaks if this fails:**
    - _pick_agent can't parse server agent listing response.
    - Agent name extraction broken.
    """
    from omnigent.chat import _pick_agent, _start_local_server, _stop_local_server

    # The archer spec's connection block uses ${OPENAI_API_KEY}, which
    # the spec parser expands at load time from the subprocess's env.
    os.environ["OPENAI_API_KEY"] = openai_judge_api_key

    port = find_free_port()
    server = _start_local_server(_ARCHER_DIR, port)
    base_url = f"http://127.0.0.1:{port}"

    try:
        wait_for_server(base_url)

        # _pick_agent auto-selects when there's only one agent.
        agent_name = _pick_agent(base_url)
        assert agent_name == "archer", (
            f"Expected _pick_agent to return 'archer', got {agent_name!r}."
        )
    finally:
        _stop_local_server(server)
