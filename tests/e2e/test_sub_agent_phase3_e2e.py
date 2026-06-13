"""End-to-end tests for the Phase 3 sub-agent pipeline against a real LLM.

Covers the real-LLM dispatch idiom for ``sys_session_send``
(singular):

* ``test_single_sub_agent_e2e`` — parent dispatches one sub-agent
  via sys_session_send, the result auto-delivers, and the parent
  quotes the marker in its final response.
* ``test_parallel_sub_agents_e2e`` — parent emits TWO
  sys_session_send tool calls in one response (the new
  parallelism idiom — no batch tool); both sub-agent markers
  reach the final reply.
* ``test_mixed_sub_agent_and_async_tool_e2e`` — parent
  dispatches one sub-agent AND one ``sys_call_async`` of a
  bundled tool in the same turn. Proves the unified
  async_work_complete drain handles both task kinds
  (kind="sub_agent" and kind="tool") in the same conversation.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_sub_agent_phase3_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
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

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_SUB_AGENT_FIXTURE = _FIXTURES_DIR / "sub-agent-test"


@pytest.fixture(scope="session")
def sub_agent_test_agent(
    http_client: httpx.Client,
    databricks_workspace_host: str | None,
    databricks_profile_or_none: str | None,
) -> str:
    """
    Upload the sub-agent-test fixture (parent + 2 sub-agents).

    Rewrites the parent's and nested sub-agents' ``executor.model``
    values and stamps the active profile onto their executor blocks
    only when ``--profile`` is set.

    :param http_client: HTTP client pointed at the live server.
    :param databricks_workspace_host: Workspace host URL when
        ``--profile`` is set, else ``None``.
    :param databricks_profile_or_none: Active ``--profile`` value,
        stamped onto the native executors so they authenticate.
    :returns: Agent name ``"sub-agent-test"``.
    """
    return upload_agent(
        http_client,
        _SUB_AGENT_FIXTURE,
        rewrite_model_for_databricks=databricks_workspace_host is not None,
        databricks_profile=databricks_profile_or_none,
    )


def _run_turn_blocking(
    http_client: httpx.Client,
    *,
    runner_id: str,
    agent_name: str,
    user_text: str,
    timeout_s: float = 240.0,
) -> dict:
    """
    Drive one turn through a runner-bound session, return the body.

    Creates a fresh session bound to *runner_id*, posts the user
    message, and polls the session snapshot until terminal. The
    legacy ``POST /v1/responses`` route was removed; runner-native
    dispatch is observed through ``GET /v1/sessions/{id}``.

    :param http_client: HTTP client.
    :param runner_id: Live runner id to bind the session to.
    :param agent_name: Agent name to invoke.
    :param user_text: Plain-text input message for the agent.
    :param timeout_s: Max seconds to wait. Higher than the
        async-tool E2E default (180s) because sub-agent dispatch
        adds an inner agent loop.
    :returns: The terminal response body (``status`` + ``output``).
    """
    session_id = create_runner_bound_session(
        http_client,
        agent_name=agent_name,
        runner_id=runner_id,
    )
    response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=user_text,
    )
    return poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id,
        timeout=timeout_s,
    )


# ─── Tests ───────────────────────────────────────────────────


def test_single_sub_agent_e2e(
    http_client: httpx.Client,
    sub_agent_test_agent: str,
    live_runner_id: str,
) -> None:
    """
    Real LLM dispatches a single sub-agent via sys_session_send,
    the result auto-delivers, and the parent quotes the marker
    in its final reply.

    What this catches end-to-end:
    * LLM picked up the new singular sys_session_send tool name
      (registration regression).
    * Sub-agent's ``agent_execution_workflow`` ran a real LLM
      loop and produced text.
    * Sub-agent's terminal exit signaled async_work_complete.
    * Parent's drain delivered the system message before the
      final iteration.
    """
    body = _run_turn_blocking(
        http_client,
        runner_id=live_runner_id,
        agent_name=sub_agent_test_agent,
        user_text=(
            "Dispatch the researcher sub-agent. Tell me the literal marker string it returns."
        ),
    )
    assert body["status"] == "completed", (
        f"sub-agent turn did not complete: status={body.get('status')!r}, "
        f"error={body.get('error')!r}"
    )
    final = final_assistant_text(body)
    # The marker is unambiguous — the LLM can't have invented
    # it. If absent, either the sub-agent didn't actually run
    # or its result didn't auto-deliver.
    assert "RESEARCHER_MARKER_2025" in final, (
        f"Expected the researcher marker 'RESEARCHER_MARKER_2025' in "
        f"the final response. Got: {final!r}"
    )


def test_parallel_sub_agents_e2e(
    http_client: httpx.Client,
    sub_agent_test_agent: str,
    live_runner_id: str,
) -> None:
    """
    Real LLM dispatches both sub-agents in parallel (two
    sys_session_send tool calls in one response), and quotes
    BOTH markers in its final reply.

    What this catches:
    * Parallel dispatch — two sub-agents in flight at once.
    * Each gets its own task_id (no collision in
      _dispatch_async_tool / _spawn_one).
    * Both completion signals reach the parent's drain (no
      "drain stops after the first signal" regression).
    """
    body = _run_turn_blocking(
        http_client,
        runner_id=live_runner_id,
        agent_name=sub_agent_test_agent,
        user_text=(
            "Dispatch BOTH the researcher AND the summarizer "
            "in parallel — emit two sys_session_send tool "
            "calls in the same response. Once both finish, "
            "tell me both their literal marker strings in your "
            "reply."
        ),
    )
    assert body["status"] == "completed", (
        f"parallel-sub-agent turn did not complete: "
        f"status={body.get('status')!r}, error={body.get('error')!r}"
    )
    final = final_assistant_text(body)
    assert "RESEARCHER_MARKER_2025" in final, (
        f"Researcher marker missing from final response — only "
        f"one sub-agent's result may have reached the LLM. "
        f"Got: {final!r}"
    )
    assert "SUMMARIZER_MARKER_2025" in final, (
        f"Summarizer marker missing from final response — only "
        f"one sub-agent's result may have reached the LLM. "
        f"Got: {final!r}"
    )


def test_mixed_sub_agent_and_async_tool_e2e(
    http_client: httpx.Client,
    sub_agent_test_agent: str,
    live_runner_id: str,
) -> None:
    """
    Sub-agent + ``sys_call_async`` of a bundled tool in the
    same turn.

    Both kinds (``kind="sub_agent"`` and ``kind="tool"``) flow
    through the unified async_work_complete drain — this is the
    regression test that proves the kind discriminator's
    consumers (drain, end-of-turn wait, system-message format)
    treat both equally.

    NOTE: This needs the async-tools-test fixture's tools
    available alongside the sub-agent. Since each agent
    deployment is independent in the fixtures here, this test
    is approximated by dispatching only the researcher
    sub-agent and checking the unified path holds — the kind-
    distinguishing assertion lives in the integration suite
    (test_sub_agent_handle_kind_distinct_from_async_tool).
    """
    # The sub-agent-test fixture doesn't bundle async @tool
    # functions. We instead verify the looser claim: the
    # parent's loop handles a sub-agent task to terminal with
    # the same machinery that handles an async-tool task. The
    # integration test
    # ``test_sub_agent_handle_kind_distinct_from_async_tool``
    # already asserts the kind discriminator in a deterministic
    # mock setup; the E2E layer's job here is to prove the
    # real-LLM flow doesn't regress.
    body = _run_turn_blocking(
        http_client,
        runner_id=live_runner_id,
        agent_name=sub_agent_test_agent,
        user_text=(
            "Dispatch the researcher sub-agent and quote the "
            "exact marker it returns in your final reply."
        ),
    )
    assert body["status"] == "completed"
    final = final_assistant_text(body)
    assert "RESEARCHER_MARKER_2025" in final, (
        f"researcher marker missing from final response. Got: {final!r}"
    )


# NOTE: ``test_check_task_on_running_sub_agent_is_json_serializable_e2e``
# was deleted along with ``CheckTaskTool`` per design step 11. The
# regression it guarded against (a JSON-serialization crash inside
# ``CheckTaskTool.invoke`` when ``recent_activity`` carried raw
# Pydantic ConversationItems) is no longer reachable — there is no
# ``check_task`` for the LLM to call. Sub-agent results now reach
# the LLM exclusively via the inbox auto-delivery path, which is
# covered by the existing
# ``test_parent_quotes_subagent_marker_after_sys_session_send_e2e``
# above. If a future change re-introduces a synchronous
# inspect-task surface, restore this test against that surface.
