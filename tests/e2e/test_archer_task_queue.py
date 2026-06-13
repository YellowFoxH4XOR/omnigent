"""End-to-end test for stateful ``@tool`` functions via ``ToolState``.

Exercises the task queue tools bundled with the archer agent:
``add_task``, ``list_tasks``, ``update_task_status``. The queue is
held in per-agent ``ToolState`` (see ``designs/TOOL_STATE.md``), so
state must survive across multiple user turns in the same
conversation.

The assertions are made against the raw tool outputs stored on the
conversation — NOT against the LLM's natural-language summary of
what it did — so LLM wording flakiness can't poison the test. As
long as the LLM actually calls the tools (which we assert by
inspecting ``function_call`` items), the state machinery itself is
what we're measuring.

What breaks if wrong:

- Schema builder doesn't skip ``tool_state``: the LLM tries to fill
  the parameter and the call returns a validation error. No
  ``function_call`` items for the tool name.
- Runner doesn't inject ``tool_state``: subprocess exits non-zero
  with TypeError; the output is an "Error:" string, not valid JSON.
- srt sandbox blocks writes to ``.tool_state``: output is the
  "Read-only file system" error string.
- ``transaction()`` broken or state not per-agent-per-conversation:
  turn 2's ``list_tasks`` output is an empty list instead of the
  tasks added in turn 1.

Runs against a real LLM via the archer agent. Requires
``--llm-api-key``.
"""

from __future__ import annotations

import json

import httpx

from tests.e2e.conftest import (
    create_runner_bound_session,
    poll_session_until_terminal,
    send_user_message_to_session,
)


def _get_tool_outputs(
    client: httpx.Client,
    session_id: str,
    tool_name: str,
) -> list[str]:
    """Return the raw string outputs of every ``tool_name`` call in order.

    Walks the conversation's function_call / function_call_output
    items and returns the outputs matching ``tool_name``. This is
    the assertion surface for state-persistence tests — the LLM's
    natural-language summary is not — because tool outputs are
    deterministic where the LLM's prose isn't.

    :param client: HTTP client pointed at the Omnigent server.
    :param session_id: The session to inspect.
    :param tool_name: Only outputs of calls to this tool are returned.
    :returns: Ordered list of raw output strings.
    """
    resp = client.get(f"/v1/sessions/{session_id}/items", params={"limit": 100})
    resp.raise_for_status()
    items = resp.json()["data"]
    calls_by_id: dict[str, dict] = {}
    for item in items:
        if item.get("type") == "function_call" and item.get("name") == tool_name:
            calls_by_id[item["call_id"]] = item
    outputs: list[str] = []
    for item in items:
        if item.get("type") == "function_call_output":
            cid = item.get("call_id")
            if cid in calls_by_id:
                outputs.append(str(item.get("output", "")))
    return outputs


def test_archer_task_queue_persists_state_across_turns(
    http_client: httpx.Client,
    archer_agent: str,
    live_runner_id: str,
) -> None:
    """Verify ToolState round-trips across four agent turns.

    Turn 1: agent adds two tasks (alpha, beta).
    Turn 2: agent lists tasks — output contains both.
    Turn 3: agent marks task 1 as done.
    Turn 4: agent lists pending — output contains beta, not alpha.

    Assertions are on the raw tool outputs stored on the
    conversation, NOT the LLM's summary, so wording variance
    can't break the test.
    """
    session_id = create_runner_bound_session(
        http_client,
        agent_name=archer_agent,
        runner_id=live_runner_id,
    )

    # ── Turn 1: add alpha then beta ──────────────────────
    response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            "Call add_task twice. First call: description='alpha'. "
            "Second call: description='beta'. Then reply 'ok'."
        ),
    )
    body = poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id,
        timeout=120,
    )
    assert body["status"] == "completed", f"Turn 1 failed: {body.get('error')}"

    add_outputs = _get_tool_outputs(http_client, session_id, "add_task")
    assert len(add_outputs) == 2, (
        f"Turn 1 should have called add_task exactly twice (LLM "
        f"compliance check), got {len(add_outputs)}. If 0, the "
        f"tool never ran; if 1, the LLM stopped short."
    )
    added = [json.loads(o) for o in add_outputs]
    descs = sorted(t["description"] for t in added)
    assert descs == ["alpha", "beta"], (
        f"Expected tasks ['alpha', 'beta'], got {descs}. LLM likely ignored the prompt."
    )
    ids = sorted(t["id"] for t in added)
    assert ids == [1, 2], (
        f"Expected IDs [1, 2] (monotonic from _empty_state), got {ids}. "
        f"A regression in the next_id bump would show up as duplicates."
    )

    # ── Turn 2: list all ─────────────────────────────────
    response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content="Call list_tasks with no arguments (status null). Reply 'listed'.",
    )
    body = poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id,
        timeout=120,
    )
    assert body["status"] == "completed", f"Turn 2 failed: {body.get('error')}"

    list_outputs = _get_tool_outputs(http_client, session_id, "list_tasks")
    assert len(list_outputs) == 1, (
        f"Turn 2 should have called list_tasks once, got {len(list_outputs)}."
    )
    listed = json.loads(list_outputs[0])
    listed_descs = sorted(t["description"] for t in listed)
    # If ToolState isn't persisting across turns, this list
    # is empty. The canary for the whole feature.
    assert listed_descs == ["alpha", "beta"], (
        f"Turn 2 list_tasks should see both tasks from turn 1, "
        f"got {listed_descs}. If [], ToolState isn't persisting; "
        f"if one entry, add_task's transaction dropped a write."
    )

    # ── Turn 3: mark task 1 done ─────────────────────────
    response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content="Call update_task_status with task_id=1 and new_status='done'. Reply 'done'.",
    )
    body = poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id,
        timeout=120,
    )
    assert body["status"] == "completed", f"Turn 3 failed: {body.get('error')}"

    upd_outputs = _get_tool_outputs(http_client, session_id, "update_task_status")
    assert len(upd_outputs) == 1, (
        f"Turn 3 should have called update_task_status once, got {len(upd_outputs)}."
    )
    updated = json.loads(upd_outputs[0])
    assert updated["id"] == 1 and updated["status"] == "done", (
        f"update_task_status should have returned the updated task "
        f"with id=1 status='done', got {updated!r}."
    )

    # ── Turn 4: list pending ─────────────────────────────
    response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content="Call list_tasks with status='pending'. Reply 'listed'.",
    )
    body = poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=response_id,
        timeout=120,
    )
    assert body["status"] == "completed", f"Turn 4 failed: {body.get('error')}"

    list_outputs_2 = _get_tool_outputs(http_client, session_id, "list_tasks")
    assert len(list_outputs_2) == 2, (
        f"Cumulative list_tasks calls should be 2, got {len(list_outputs_2)}."
    )
    pending = json.loads(list_outputs_2[-1])
    pending_descs = sorted(t["description"] for t in pending)
    assert pending_descs == ["beta"], (
        f"Turn 4 list_tasks(status='pending') should see only "
        f"'beta' (alpha was marked done), got {pending_descs}. "
        f"If ['alpha', 'beta'], update_task_status didn't "
        f"persist; if [], the filter is broken or state was lost."
    )
