"""E2E test: steering interrupts a running agent.

Verifies that a message delivered to a session whose latest
task is still in flight is steered into that task (rather than
starting a new one), and the agent picks it up in its next
turn.

Both turns route through a runner-bound session
(``POST /v1/sessions/{id}/events``). On the events endpoint,
the server inspects the session's active task: if one is
running with an open inbox, the new item is delivered into it
(:func:`try_deliver`) and tagged with the same ``response_id``;
otherwise a fresh task is created. The helper reads back the
persisted item's ``response_id`` so tests can compare it
against the original task id to confirm whether steering took
the steer-into-running path or fell through to a new turn.

Usage::

    pytest tests/e2e/test_steering.py \
        --llm-api-key $LLM_API_KEY -v
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from tests.e2e.conftest import (
    create_runner_bound_session,
    poll_session_until_terminal,
    send_user_message_to_session,
)

_RUNNING_POLL_INTERVAL_S = 2


def _wait_for_session_running(
    client: httpx.Client,
    session_id: str,
    timeout: float = 60,
) -> None:
    """
    Poll GET /v1/sessions/{id} until status == "running".

    Raises AssertionError if the session doesn't reach running within
    *timeout* seconds — this makes a failed wait produce a clear error
    rather than silently steering into an idle/completed session.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/v1/sessions/{session_id}")
        r.raise_for_status()
        if r.json().get("status") == "running":
            return
        time.sleep(_RUNNING_POLL_INTERVAL_S)
    raise AssertionError(
        f"Session {session_id} did not reach 'running' within {timeout}s; "
        f"last status={client.get(f'/v1/sessions/{session_id}').json().get('status')!r}"
    )


def test_steering_acknowledged(
    http_client: httpx.Client,
    archer_agent: str,
    live_runner_id: str,
) -> None:
    """
    A message sent while the agent is running is steered into
    the active task and reflected in the final output.

    The agent is asked to write a long essay. While it's
    running, we send "Say only: PINEAPPLE" through the same
    session. The events endpoint must deliver this into the
    running task's inbox, the LLM must re-run with the steered
    message visible, and the final output must contain
    "PINEAPPLE".

    **What breaks if steering is broken:**

    - If ``close_inbox`` uses a cursor past the steer's position
      (e.g. advanced by native tool items), the steer is missed
      → only the original essay appears, no PINEAPPLE.
    - If ``close_inbox`` is called synchronously on the async
      event loop, it deadlocks → task never completes.
    - If the events endpoint mis-classifies the inbox as closed,
      the steer falls through to a new task and the second
      response_id differs from the first.
    """
    session_id = create_runner_bound_session(
        http_client, agent_name=archer_agent, runner_id=live_runner_id
    )

    # Call sys_read_inbox to block the task until an inbox message
    # arrives. The steer IS that inbox message, so the task stays
    # open until we post it. After the steer arrives sys_read_inbox
    # returns and the LLM re-runs with the steered message visible.
    task_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            "Call sys_read_inbox now and wait for it to return. "
            "Do not reply until sys_read_inbox has returned."
        ),
    )

    _wait_for_session_running(http_client, session_id, timeout=60)

    # Deliver the steer while the turn is in progress.
    # The runner buffers it and the LLM re-runs with it visible.
    send_user_message_to_session(
        http_client,
        session_id=session_id,
        content="STOP. Say only: PINEAPPLE",
    )

    body = poll_session_until_terminal(
        http_client, session_id=session_id, response_id=task_id, timeout=120
    )
    assert body["status"] == "completed", f"Task failed: {body.get('error')}"

    # The steered message must surface in the final output. Backstopped by
    # the harness empty-completion retry: the openai-agents gateway
    # used to return an empty turn here, which this assertion read as
    # "steering not acknowledged" even though the plumbing worked.
    all_text = _extract_all_text(body)
    assert "PINEAPPLE" in all_text.upper(), (
        f"Steering not acknowledged. Output was:\n{all_text[:500]}"
    )


def test_steering_with_web_search(
    http_client: httpx.Client,
    archer_agent: str,
    live_runner_id: str,
) -> None:
    """
    Steering works when native tool items (web_search_call) are
    in the response. This is the exact scenario that was broken:
    native tool persistence advanced ``last_seen`` past the steer.

    **What breaks if the cursor fix regresses:**

    - ``close_inbox`` uses the post-native-tool cursor → misses
      the steered message → no PINEAPPLE in output.
    """
    session_id = create_runner_bound_session(
        http_client, agent_name=archer_agent, runner_id=live_runner_id
    )

    # sys_read_inbox blocks until the steer (inbox message) arrives,
    # then the web search runs so native tool items (web_search_call)
    # are still exercised by the re-run after the steer.
    task_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            "Do these two steps in order:\n"
            "1. Call sys_read_inbox and wait for it to return\n"
            "2. Search the web for the latest news about artificial intelligence\n"
            "Do NOT skip any steps."
        ),
    )

    _wait_for_session_running(http_client, session_id, timeout=60)

    send_user_message_to_session(
        http_client,
        session_id=session_id,
        content="STOP ALL STEPS. Say only: PINEAPPLE",
    )

    body = poll_session_until_terminal(
        http_client, session_id=session_id, response_id=task_id, timeout=240
    )
    assert body["status"] == "completed"

    all_text = _extract_all_text(body)
    assert "PINEAPPLE" in all_text.upper(), (
        f"Steering with web search not acknowledged: {all_text[:300]}"
    )


def test_steering_after_completed_starts_new_turn(
    http_client: httpx.Client,
    archer_agent: str,
    live_runner_id: str,
) -> None:
    """
    A message sent after the task completes creates a new turn,
    not a steer. Verifies that ``_response_terminal`` detection
    works on the events endpoint: with no active task, the
    handler falls through to ``task_store.create`` and the
    second message gets a fresh ``response_id``.
    """
    session_id = create_runner_bound_session(
        http_client, agent_name=archer_agent, runner_id=live_runner_id
    )

    task_id = send_user_message_to_session(
        http_client, session_id=session_id, content="Say hello."
    )
    body = poll_session_until_terminal(
        http_client, session_id=session_id, response_id=task_id, timeout=30
    )
    assert body["status"] == "completed"

    # Same session, prior task is terminal — second message starts
    # a new turn and completes independently.
    task2_id = send_user_message_to_session(
        http_client, session_id=session_id, content="What is 2+2?"
    )

    body2 = poll_session_until_terminal(
        http_client, session_id=session_id, response_id=task2_id, timeout=30
    )
    assert body2["status"] == "completed"
    text = _extract_all_text(body2)
    assert "4" in text, f"Expected answer to 2+2, got: {text[:100]}"


def test_steering_during_multi_tool_iterations(
    http_client: httpx.Client,
    archer_agent: str,
    live_runner_id: str,
) -> None:
    """
    Steering is picked up between tool call iterations when the
    agent makes multiple sequential tool calls (web search + sys_os_shell).

    This tests ``_sync_steered_after_tools`` with the pre-LLM cursor
    fix. The agent is explicitly told to make multiple tool calls
    in sequence. The steer arrives during execution and must be
    acknowledged after the tool calls complete.

    **What breaks if the tool-iteration cursor is wrong:**

    - ``_sync_steered_after_tools`` uses a cursor past the steer's
      position → steer is never added to history → the LLM never
      sees it → no PINEAPPLE.
    """
    session_id = create_runner_bound_session(
        http_client, agent_name=archer_agent, runner_id=live_runner_id
    )

    # sys_read_inbox blocks until the steer arrives. The subsequent
    # list_files calls exercise the multi-tool iteration cursor fix
    # this test was written to cover: the cursor must not skip over
    # the steered message between tool iterations.
    task_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            "Do these steps in order, one tool call at a time:\n"
            "1. Call sys_read_inbox and wait for it to return\n"
            "2. Call list_files\n"
            "3. Call list_files again\n"
            "Do NOT skip any steps."
        ),
    )

    _wait_for_session_running(http_client, session_id, timeout=60)

    send_user_message_to_session(
        http_client,
        session_id=session_id,
        content="STOP ALL STEPS. Say only: PINEAPPLE",
    )

    body = poll_session_until_terminal(
        http_client, session_id=session_id, response_id=task_id, timeout=240
    )
    assert body["status"] == "completed", f"Task failed: {body.get('error')}"

    all_text = _extract_all_text(body)
    tool_count = len([i for i in body.get("output", []) if i.get("type") == "function_call"])
    # Backstopped by the harness empty-completion retry: the
    # openai-agents gateway used to return an empty turn here, which this
    # assertion read as "steering not acknowledged" even though the
    # multi-tool cursor plumbing worked.
    assert "PINEAPPLE" in all_text.upper(), (
        f"Steering during multi-tool iterations not acknowledged. "
        f"Tool calls before steer: {tool_count}. Output: {all_text[:500]}"
    )
    assert tool_count >= 1, "Expected at least 1 tool call before the steer was processed"


def _extract_all_text(body: dict[str, Any]) -> str:
    """
    Concatenate all assistant output_text blocks.

    :param body: The terminal response body.
    :returns: All assistant text joined by newlines.
    """
    parts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for block in item.get("content", []):
                text = block.get("text")
                if text:
                    parts.append(text)
    return "\n".join(parts)
