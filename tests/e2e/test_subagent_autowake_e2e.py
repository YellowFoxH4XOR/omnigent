"""End-to-end test for sub-agent auto-wake against a real LLM.

When a sub-agent finishes, the runner delivers its result to the parent's
inbox AND posts a ``[System: ... waiting in inbox]`` wake notice to the
parent's event stream, so an idle orchestrator takes a continuation turn and
surfaces the result — **without the user sending another message**.

Before the auto-wake fix, the parent dispatched a sub-agent, ended its turn,
and then sat idle until the next user message ("the orchestrator doesn't know
its sub-agent finished"). This test reproduces that scenario end-to-end: it
sends exactly ONE user message (the dispatch) and then asserts — purely from
later polling, with no further input — that the wake notice and the
sub-agent's marker appear in the conversation.

The wake notice substring ``waiting in inbox`` is produced ONLY by the
auto-wake path (``_format_subagent_wake_notice``); it is distinct from the
``sys_read_inbox`` drain message. So its presence is an auto-wake-specific
signal that is robust to how the parent LLM manages its turns.

Excluded from the default ``pytest`` run via ``--ignore=tests/e2e``. Invoke
against the OSS gateway with::

    pytest tests/e2e/test_subagent_autowake_e2e.py \\
        --profile oss \\
        --llm-api-key "$(databricks auth token --profile oss | jq -r .access_token)" -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import (
    create_runner_bound_session,
    poll_session_until_terminal,
    send_user_message_to_session,
    upload_agent,
)
from tests.e2e.helpers import POLL_INTERVAL_S

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_SUB_AGENT_FIXTURE = _FIXTURES_DIR / "sub-agent-test"

# The auto-wake notice (omnigent/runner/app.py::_format_subagent_wake_notice)
# is the ONLY place this substring is emitted — the sys_read_inbox drain
# message uses different wording ("returned:"). So finding it in the parent's
# conversation proves the wake POST fired, not synchronous dispatch.
_WAKE_NOTICE_SIGNATURE = "waiting in inbox"
_RESEARCHER_MARKER = "RESEARCHER_MARKER_2025"


@pytest.fixture(scope="session")
def sub_agent_test_agent(
    http_client: httpx.Client,
    databricks_workspace_host: str | None,
    databricks_profile_or_none: str | None,
) -> str:
    """
    Upload the sub-agent-test fixture (parent + researcher/summarizer).

    :param http_client: HTTP client pointed at the live server.
    :param databricks_workspace_host: Workspace host URL when ``--profile``
        is set (rewrites ``model:`` values to Databricks names), else ``None``.
    :param databricks_profile_or_none: Active ``--profile``, stamped onto
        the native executors so they authenticate to the gateway.
    :returns: Agent name ``"sub-agent-test"``.
    """
    return upload_agent(
        http_client,
        _SUB_AGENT_FIXTURE,
        rewrite_model_for_databricks=databricks_workspace_host is not None,
        databricks_profile=databricks_profile_or_none,
    )


def _session_items_blob(http_client: httpx.Client, session_id: str) -> str:
    """
    Return all items in a session snapshot as one JSON string.

    Serializing the whole item list lets the caller substring-search for an
    unambiguous marker / notice wherever it lands (assistant text, a tool
    result, or a framework user message) without coupling to item shapes.

    :param http_client: HTTP client pointed at the live server.
    :param session_id: Session/conversation id, e.g. ``"conv_abc123"``.
    :returns: ``json.dumps`` of the snapshot's ``items`` list.
    """
    resp = http_client.get(f"/v1/sessions/{session_id}")
    resp.raise_for_status()
    return json.dumps(resp.json().get("items", []))


def _count_wake_notices(http_client: httpx.Client, session_id: str) -> int:
    """
    Count auto-wake notices in a session snapshot.

    Each wake posts one ``waiting in inbox`` notice, so the count lets a
    multi-round test prove a later round produced its own wake.

    :param http_client: HTTP client pointed at the live server.
    :param session_id: Session/conversation id, e.g. ``"conv_abc123"``.
    :returns: Number of ``waiting in inbox`` occurrences across all items.
    """
    return _session_items_blob(http_client, session_id).count(_WAKE_NOTICE_SIGNATURE)


def test_subagent_completion_auto_wakes_idle_parent(
    http_client: httpx.Client,
    live_runner_id: str,
    sub_agent_test_agent: str,
) -> None:
    """
    Dispatching a sub-agent then sending nothing else still surfaces its
    result, because the parent is auto-woken when the sub-agent completes.

    Flow:
    1. One user message tells the parent to dispatch the researcher and end
       its turn (no busy-wait).
    2. The dispatch turn goes terminal — the marker is NOT expected yet,
       since ``sys_session_send`` is async and the sub-agent runs after the
       parent's turn ends.
    3. With NO further user input, the sub-agent completes, the runner posts
       the wake notice, and the parent takes a continuation turn that reads
       the inbox and quotes the marker.

    Without the auto-wake fix, step 3 never happens: the parent stays idle,
    the wake notice is never posted, and the marker never appears — the
    poll below times out and the test fails.
    """
    session_id = create_runner_bound_session(
        http_client,
        agent_name=sub_agent_test_agent,
        runner_id=live_runner_id,
    )
    dispatch_response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            "Dispatch the researcher sub-agent with exactly ONE sys_session_send "
            "call, then end your turn with a one-line acknowledgement like "
            "'dispatched, waiting'. Do NOT call sys_read_inbox or any other tool "
            "in this turn. Later, when you are notified the result is ready, read "
            "it and quote the researcher's literal marker string verbatim."
        ),
    )

    # The dispatch turn goes terminal on its own (the sub-agent runs async).
    poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=dispatch_response_id,
        timeout=180,
    )

    # From here on we send NOTHING. The only way the wake notice and the
    # marker can appear is the auto-wake continuation turn fired by the
    # sub-agent's completion.
    deadline = time.monotonic() + 240
    wake_seen = False
    marker_seen = False
    while time.monotonic() < deadline:
        blob = _session_items_blob(http_client, session_id)
        wake_seen = wake_seen or _WAKE_NOTICE_SIGNATURE in blob
        marker_seen = _RESEARCHER_MARKER in blob
        if wake_seen and marker_seen:
            break
        time.sleep(POLL_INTERVAL_S)

    # The wake notice is emitted ONLY by the auto-wake POST. Its absence means
    # the parent was never woken after the sub-agent finished — the exact bug
    # this feature fixes.
    assert wake_seen, (
        f"No auto-wake notice ({_WAKE_NOTICE_SIGNATURE!r}) appeared in session "
        f"{session_id} after the dispatch turn ended. The parent was never woken "
        f"by the sub-agent's completion."
    )
    # The marker proves the sub-agent actually ran and its result reached the
    # parent through the woken continuation turn (not just that a turn fired).
    assert marker_seen, (
        f"Researcher marker {_RESEARCHER_MARKER!r} never surfaced in session "
        f"{session_id}. The parent was woken (notice present) but did not read "
        f"the inbox / quote the result."
    )


def test_subagent_completion_auto_wakes_parent_on_a_second_round(
    http_client: httpx.Client,
    live_runner_id: str,
    sub_agent_test_agent: str,
) -> None:
    """
    Re-dispatching the SAME sub-agent in a second round wakes the parent again.

    This is a COARSE real-LLM CUJ for the multi-round auto-wake path: round 1
    dispatches the researcher and the parent is auto-woken; round 2
    re-dispatches and the parent must be auto-woken AGAIN, asserted by the
    wake-notice count strictly increasing.

    IMPORTANT — coverage boundary (do not rely on this test for the
    stuck-flag fix): this e2e does NOT deterministically recreate the
    stuck-debounce-flag state the fix targets. The round-2 user message
    starts a FRESH parent turn, and turn start is exactly where
    ``_subagent_wake_pending`` is cleared — so the flag is reset before the
    round-2 child completes, regardless of the fix. The bug only manifests
    when a wake is consumed MID-TURN as an injection (no turn-start clear),
    and real-LLM timing cannot reliably force a child to complete during such
    a turn (nor control whether the parent drains ``sys_read_inbox`` first).
    Forcing it with sleeps/prompts would be "fire and hope" flakiness, so this
    test stays a coarse CUJ and could pass even with the bug present.

    The DETERMINISTIC guards for the stuck-flag fix live in the unit tests in
    ``tests/runner/test_app_sessions_native.py``:
    - ``test_parent_idle_with_stuck_wake_flag_posts_recovery_wake`` — flag
      stuck + NON-EMPTY inbox at idle → recovery wake re-armed.
    - ``test_parent_idle_with_stuck_wake_flag_and_drained_inbox_clears_flag``
      — flag stuck + EMPTY inbox at idle (drained mid-turn) → flag cleared so
      the next completion wakes instead of stranding.
    Those drive the mid-turn-consumed path directly with a blocking harness;
    this e2e only confirms the end-to-end golden path still wakes per round.
    """
    session_id = create_runner_bound_session(
        http_client,
        agent_name=sub_agent_test_agent,
        runner_id=live_runner_id,
    )

    _dispatch_instruction = (
        "Dispatch the researcher sub-agent with exactly ONE sys_session_send "
        "call, then end your turn with a one-line acknowledgement like "
        "'dispatched, waiting'. Do NOT call sys_read_inbox or any other tool "
        "in this turn. Later, when you are notified the result is ready, read "
        "it and quote the researcher's literal marker string verbatim."
    )

    # ── Round 1 ──────────────────────────────────────────────────────────
    round1_response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=_dispatch_instruction,
    )
    poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=round1_response_id,
        timeout=180,
    )

    deadline = time.monotonic() + 240
    round1_wakes = 0
    marker_seen = False
    while time.monotonic() < deadline:
        round1_wakes = _count_wake_notices(http_client, session_id)
        marker_seen = _RESEARCHER_MARKER in _session_items_blob(http_client, session_id)
        if round1_wakes >= 1 and marker_seen:
            break
        time.sleep(POLL_INTERVAL_S)

    assert round1_wakes >= 1 and marker_seen, (
        f"Round 1 did not auto-wake the parent in session {session_id} "
        f"(wakes={round1_wakes}, marker_seen={marker_seen}); cannot test round 2."
    )

    # ── Round 2: re-dispatch the SAME sub-agent; parent must wake AGAIN ──
    # NOTE: this fresh user message starts a new parent turn, which clears
    # _subagent_wake_pending at turn start — so it does NOT reliably reproduce
    # the mid-turn-consumed stuck-flag state the fix targets (see docstring).
    # The deterministic stuck-flag coverage is in the unit tests named there.
    # This round only asserts the coarse CUJ: a second round still auto-wakes.
    round2_response_id = send_user_message_to_session(
        http_client,
        session_id=session_id,
        content=(
            "Now dispatch the researcher sub-agent AGAIN with exactly ONE "
            "sys_session_send call, then end your turn with a one-line "
            "acknowledgement. Do NOT call sys_read_inbox in this turn. When "
            "you are later notified the new result is ready, read it and quote "
            "the researcher's literal marker string verbatim."
        ),
    )
    poll_session_until_terminal(
        http_client,
        session_id=session_id,
        response_id=round2_response_id,
        timeout=180,
    )

    # From here we send NOTHING. A higher wake count can only come from the
    # round-2 completion auto-waking the parent — what the fix restores.
    deadline = time.monotonic() + 240
    round2_wakes = round1_wakes
    while time.monotonic() < deadline:
        round2_wakes = _count_wake_notices(http_client, session_id)
        if round2_wakes > round1_wakes:
            break
        time.sleep(POLL_INTERVAL_S)

    assert round2_wakes > round1_wakes, (
        f"No NEW auto-wake notice appeared for the second round in session "
        f"{session_id} (round1_wakes={round1_wakes}, round2_wakes={round2_wakes}). "
        f"The second-round sub-agent completion did not wake the parent — the "
        f"stranding bug (debounce flag stuck from a mid-turn-consumed wake)."
    )
