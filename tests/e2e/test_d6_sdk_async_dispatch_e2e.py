"""
E2E for the SDK-side async client-tool lifecycle dispatched
via ``sys_call_async``.

Proves the python-client SDK drives the full async path
end-to-end without any caller bookkeeping:

1. SDK exposes ``@tool``-decorated client tools on the wire.
2. Real LLM dispatches one via
   ``sys_call_async(tool=..., args=...)``.
3. Server's :meth:`SysCallAsyncTool.dispatch_async` creates a
   ``kind="client_tool"`` task, registers a pending_tool_call
   keyed to a synthesized call_id, starts
   ``client_tool_workflow`` parked on
   ``CLIENT_TOOL_RESULT_TOPIC``, and synthesizes a
   ``function_call(action_required)`` SSE event.
4. SDK's action_required handler fires
   ``_execute_and_patch`` to run the tool body locally and
   PATCH ``tool_results`` back when it completes.
5. Server's PATCH handler completes the pending row and
   bridges to ``CLIENT_TOOL_RESULT_TOPIC``; the holder
   workflow wakes and sends ``async_work_complete`` to the
   parent.
6. Parent's drain renders ``[System: task X (client_tool)
   completed]\\n<body>`` as a user message.
7. LLM reads the system message and replies ``ANSWER:<body>``.
8. Test asserts the body text round-tripped through both the
   tool and the drain.

Excluded from default ``pytest`` runs via
``--ignore=tests/e2e``. Invoke with::

    pytest tests/e2e/test_d6_sdk_async_dispatch_e2e.py \\
        --llm-api-key "$(cat /tmp/mykey)" -v
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from omnigent_client import OmnigentClient
from omnigent_client._events import (
    MessageDone,
    ResponseCompleted,
    ResponseFailed,
    ResponseIncomplete,
    TextDelta,
)
from omnigent_client.tools import build_tool_handler, tool

from tests.e2e.conftest import upload_agent

_FIXTURES_DIR = Path(__file__).resolve().parents[1] / "_fixtures" / "agents"
_FIXTURE = _FIXTURES_DIR / "d6-sdk-async-dispatch-test"

# Marker the @tool body returns. The agent's AGENTS.md instructs
# the LLM to echo the marker back via ``ANSWER:<body>`` after
# the drain delivers it as a system message — finding the
# marker in the LLM's final assistant text proves the entire
# loop closed: SDK dispatch → server drain → LLM reads system
# message → SDK PATCH → drain delivery.
_MARKER = "D6_SDK_ASYNC_LIFECYCLE_OK_77"


@pytest.fixture(scope="session")
def d6_test_agent(http_client: httpx.Client) -> str:
    """Upload the D6 E2E fixture."""
    return upload_agent(http_client, _FIXTURE)


# Tool body — the SDK runs this in an asyncio.Task when the
# server's :meth:`SysCallAsyncTool.dispatch_async` synthesizes
# a ``function_call(action_required)`` SSE event for it.
# Returns ``value`` verbatim so the LLM's ``ANSWER:<body>`` can
# be matched against the input marker.
@tool
async def compute(value: str) -> str:
    """Echo the input string back asynchronously.

    Args:
        value: Marker to echo. Test asserts this is what the
            LLM ultimately replies with.
    """
    # Tiny await so the body is visibly async (not just sync
    # masquerading) — proves the asyncio.Task path is what
    # ran, not an inline-execute fallback.
    await asyncio.sleep(0.05)
    return value


@pytest.mark.asyncio
async def test_sdk_async_client_tool_completes_round_trip(
    live_server: str,
    d6_test_agent: str,
) -> None:
    """
    Full SDK-driven async-client-tool lifecycle, end-to-end.

    Failure modes this test catches:

    - LLM doesn't dispatch via ``sys_call_async`` (e.g. calls
      ``compute`` directly) → call lands in
      ``pending_client_calls`` and runs synchronously, the
      test deadlocks waiting for the system message that the
      sync path never produces.
    - Server's :func:`_dispatch_client_tool_async` doesn't
      register the pending_tool_call → the SDK PATCHes
      ``tool_results`` and the server returns 404 → the
      holder workflow runs out the 1h cap (test would hit
      its own timeout first).
    - PATCH-handler bridge to ``CLIENT_TOOL_RESULT_TOPIC``
      regresses → ``client_tool_workflow`` never wakes →
      drain never delivers ``completed`` → LLM has nothing
      to ANSWER with.
    - Server-side audit-fix-#1 routing regresses → drain
      message lands on the wrong agent → LLM never sees it.
    """
    handler = build_tool_handler([compute])

    async with OmnigentClient(base_url=live_server) as client:
        # Drive the stream to completion. Collect events so the
        # test can assert on terminal status + the assistant
        # message body that contains the marker.
        final_text_chunks: list[str] = []
        terminal_status: str | None = None
        failure_diag: str | None = None

        async for event in client.responses.stream(
            model=d6_test_agent,
            input=f"Compute on the value {_MARKER!r}.",
            tool_handler=handler,
        ):
            if isinstance(event, TextDelta):
                # Server streams assistant text incrementally as
                # output_text deltas; the corresponding
                # MessageDone fires with empty content (the
                # OpenAI streaming convention). Accumulate
                # deltas to capture the LLM's actual output.
                final_text_chunks.append(event.delta)
            elif isinstance(event, MessageDone):
                # Empty under the OpenAI streaming convention,
                # but harmless to also catch any non-streamed
                # content blocks for forward-compat.
                for block in event.content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text = block.get("text") or ""
                        if isinstance(text, str) and text:
                            final_text_chunks.append(text)
            elif isinstance(event, ResponseCompleted):
                terminal_status = "completed"
            elif isinstance(event, ResponseFailed):
                terminal_status = "failed"
                err = event.response.error
                failure_diag = repr(err)[:600] if err is not None else "no error info"
            elif isinstance(event, ResponseIncomplete):
                terminal_status = "incomplete"

    assert terminal_status == "completed", (
        f"D6 lifecycle should complete cleanly; got "
        f"terminal_status={terminal_status!r}, "
        f"failure_diag={failure_diag!r}, "
        f"final_text_chunks={final_text_chunks!r}"
    )

    # The LLM's reply should contain the marker (per the agent
    # AGENTS.md instructions: ANSWER:<body> where <body> is the
    # system message body, which is the tool's return value,
    # which is the marker). Streamed deltas are concatenated
    # without separators — they're token-level fragments of one
    # continuous text, not separate messages.
    joined = "".join(final_text_chunks)
    assert _MARKER in joined, (
        f"D6 lifecycle round-trip failed: marker {_MARKER!r} not "
        f"found in any assistant message text. "
        f"final_text_chunks={final_text_chunks!r}. "
        f"\nIf the test hangs / times out instead of failing here, "
        f"the SDK probably didn't fire ``_execute_and_patch`` for "
        f"the synthesized action_required event — see the "
        f"action_required branch in omnigent_client/_responses.py "
        f"and ``_dispatch_client_tool_async`` server-side."
    )
