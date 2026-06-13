"""Tests for executor event serialization (roundtrip fidelity).

Covers: event roundtrip serialization for all executor event types.
These events are serialized to JSON for DBOS @step caching and must
survive serialize→deserialize roundtrips without data loss.
"""

from __future__ import annotations

import pytest

from omnigent.runtime.executors import (
    ContextWindowExceeded,
    ExecutorError,
    NativeToolOutput,
    ReasoningChunk,
    TextChunk,
    ToolCallRequested,
    TurnCancelled,
    TurnComplete,
    dict_to_event,
    event_to_dict,
)

# ── Event serialization roundtrip ──────────────────────────────


@pytest.mark.parametrize(
    "event",
    [
        TextChunk(text="Hello"),
        ReasoningChunk(delta="thinking...", event_type="reasoning_text"),
        NativeToolOutput(item={"type": "web_search_call", "id": "ws_1"}),
        ToolCallRequested(
            call_id="call_abc",
            name="get_weather",
            arguments={"city": "London"},
        ),
        TurnComplete(text="Done."),
        TurnComplete(text=None),
        TurnCancelled(reason="user_cancelled", partial_text="I was saying"),
        TurnCancelled(reason="user_cancelled", partial_text=None),
        ContextWindowExceeded(max_tokens=128000, actual_tokens=142000),
        ExecutorError(message="auth failed", code="401"),
        ExecutorError(message="unknown error", code=None),
        ExecutorError(message="turn failed", code=None, retryable=True),
    ],
    ids=[
        "text_chunk",
        "reasoning_chunk",
        "native_tool_output",
        "tool_call_requested",
        "turn_complete_with_text",
        "turn_complete_no_text",
        "turn_cancelled_with_partial_text",
        "turn_cancelled_no_partial_text",
        "context_window_exceeded",
        "executor_error_with_code",
        "executor_error_no_code",
        "executor_error_retryable",
    ],
)
def test_event_serialization_roundtrip(
    event: TextChunk
    | ReasoningChunk
    | NativeToolOutput
    | ToolCallRequested
    | TurnComplete
    | TurnCancelled
    | ContextWindowExceeded
    | ExecutorError,
) -> None:
    """
    Every executor event type must survive a serialize→deserialize roundtrip.

    This is critical for DBOS @step caching: events are serialized to JSON
    on first execution and deserialized on replay. A broken roundtrip means
    cached events lose data on crash recovery.

    :param event: The executor event to roundtrip.
    """
    serialized = event_to_dict(event)

    # Serialized form must be a dict with a "type" key matching the class name.
    assert isinstance(serialized, dict)
    assert serialized["type"] == type(event).__name__

    deserialized = dict_to_event(serialized)

    # Roundtrip must produce an identical event. If not, the @step cache
    # would return corrupted data on replay.
    assert deserialized == event


def test_event_to_dict_unknown_type_raises() -> None:
    """
    ``event_to_dict`` must reject unknown event types with a clear error.
    """
    with pytest.raises(ValueError, match="Unknown event type"):
        event_to_dict("not an event")  # type: ignore[arg-type]


def test_dict_to_event_unknown_type_raises() -> None:
    """
    ``dict_to_event`` must reject dicts with unknown type keys.
    """
    with pytest.raises(ValueError, match="Unknown event type"):
        dict_to_event({"type": "FakeEventType", "data": 42})


# ── ToolCallObserved roundtrip (separate — more fields) ─────────


def test_tool_call_observed_serialization_roundtrip() -> None:
    """
    ToolCallObserved has 6 fields — verify all survive the roundtrip.
    """
    from omnigent.runtime.executors import ToolCallObserved

    event = ToolCallObserved(
        call_id="call_xyz",
        name="Bash",
        arguments={"command": "ls -la"},
        result="/home/user\n",
        status="success",
        duration_ms=342.1,
    )
    serialized = event_to_dict(event)
    deserialized = dict_to_event(serialized)

    # All 6 fields must match after roundtrip. duration_ms is a float —
    # JSON preserves it, but a broken serializer might lose precision.
    assert deserialized == event
