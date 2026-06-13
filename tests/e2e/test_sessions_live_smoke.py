"""Tiny live smoke for the maintained ``/v1/sessions`` path.

This intentionally is not a replacement for the old broad
``test_dispatch_fork_e2e.py`` matrix. It proves one representative
real-harness / real-LLM turn still works through the production
session flow:

1. upload an Omnigent harness-backed agent,
2. create and runner-bind a session,
3. POST a user message to ``/v1/sessions/{id}/events``,
4. poll the session snapshot to idle, and
5. assert the model reply contains a requested marker.

Run explicitly with Databricks credentials, for example::

    .venv/bin/python -m pytest tests/e2e/test_sessions_live_smoke.py \
        --profile <your-profile> --llm-api-key <your-token> -v
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import (
    create_runner_bound_session,
    poll_session_until_terminal,
    send_user_message_to_session,
    upload_agent,
)
from tests.e2e.helpers import final_assistant_text


def _write_openai_agents_smoke_yaml(tmp_path: Path) -> Path:
    """Create a minimal harness-backed Omnigent YAML bundle directory."""
    agent_dir = tmp_path / "sessions-live-smoke-agent"
    agent_dir.mkdir()
    (agent_dir / "sessions-live-smoke-agent.yaml").write_text(
        "\n".join(
            [
                "name: sessions-live-smoke-agent",
                "description: Minimal live smoke for /v1/sessions harness dispatch.",
                "executor:",
                "  harness: openai-agents",
                "  model: gpt-5.4",
                "prompt: |",
                "  You are a terse smoke-test assistant.",
                "  Follow the user's instruction exactly.",
                "",
            ]
        )
    )
    return agent_dir


@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_live_sessions_path_round_trips_through_openai_agents_harness(
    http_client: httpx.Client,
    live_runner_id: str,
    databricks_workspace_host: str | None,
    tmp_path: Path,
) -> None:
    """A real harness-backed session turn reaches the LLM and returns text.

    Uses a tiny one-off Omnigent YAML with
    ``executor.harness: openai-agents`` because that harness is pure
    Python and does not require an external CLI binary. The test goes
    through the maintained sessions route rather than the deprecated
    responses dispatch-fork path.
    """
    agent_name = upload_agent(
        http_client,
        _write_openai_agents_smoke_yaml(tmp_path),
        rewrite_model_for_databricks=databricks_workspace_host is not None,
    )
    session_id = create_runner_bound_session(
        http_client,
        agent_name=agent_name,
        runner_id=live_runner_id,
    )

    marker = "SESSIONS_LIVE_SMOKE_OK"
    response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            f"Reply with exactly the literal string {marker} and nothing else. "
            "Do not call tools or sub-agents."
        ),
    )
    body = poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id,
        timeout=180,
    )

    assert body["status"] == "completed", (
        f"sessions live smoke failed: status={body['status']!r}, "
        f"error={body.get('error')!r}, output={body.get('output')!r}"
    )
    text = final_assistant_text(body)
    assert marker in text, f"marker {marker!r} missing from assistant text: {text!r}"
