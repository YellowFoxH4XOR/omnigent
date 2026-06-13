"""Executor event types, context, and serialization."""

from omnigent.runtime.executors.base import (
    ContextWindowExceeded,
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    NativeToolOutput,
    ReasoningChunk,
    TextChunk,
    ToolCallInProgress,
    ToolCallObserved,
    ToolCallRequested,
    ToolResult,
    TurnCancelled,
    TurnComplete,
    dict_to_event,
    event_to_dict,
)

__all__ = [
    "ContextWindowExceeded",
    "Executor",
    "ExecutorContext",
    "ExecutorError",
    "ExecutorEvent",
    "NativeToolOutput",
    "ReasoningChunk",
    "TextChunk",
    "ToolCallInProgress",
    "ToolCallObserved",
    "ToolCallRequested",
    "ToolResult",
    "TurnCancelled",
    "TurnComplete",
    "dict_to_event",
    "event_to_dict",
]
