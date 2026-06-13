"""
Inner :class:`Executor` adapter for the Databricks Supervisor API.

Harness-side bridge for the supervisor: lets
harness HTTP drives the Databricks Agent Bricks
Supervisor API through the same subprocess-RPC contract as
``claude-sdk`` / ``codex`` / ``pi`` / ``openai-agents``. Delegates
to :class:`omnigent.inner.databricks_supervisor_gateway.SupervisorExecutor`
to keep the gateway-protocol logic in one place, and translates
its events to the inner :mod:`omnigent.inner.executor`
vocabulary that :class:`ExecutorAdapter` expects.

Env var contract and integration overview live in
``designs/DATABRICKS_SUPERVISOR_API_INTEGRATION.md`` and
:mod:`omnigent.inner.databricks_supervisor_harness` respectively.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NoReturn

from omnigent.inner.databricks_supervisor_gateway import (
    SupervisorExecutor as _RuntimeSupervisorExecutor,
)
from omnigent.inner.executor import (
    EnqueuedContent,
    Executor,
    ExecutorConfig,
    ExecutorError,
    ExecutorEvent,
    Message,
    TextChunk,
    ToolArgs,
    ToolCallComplete,
    ToolCallMetadata,
    ToolCallRequest,
    ToolCallStatus,
    ToolSpec,
    TurnComplete,
)
from omnigent.runtime.executors.base import (
    ExecutorContext,
)
from omnigent.runtime.executors.base import (
    ExecutorError as RuntimeExecutorError,
)
from omnigent.runtime.executors.base import (
    TextChunk as RuntimeTextChunk,
)
from omnigent.runtime.executors.base import (
    ToolCallObserved as RuntimeToolCallObserved,
)
from omnigent.runtime.executors.base import (
    ToolCallRequested as RuntimeToolCallRequested,
)
from omnigent.runtime.executors.base import (
    TurnComplete as RuntimeTurnComplete,
)
from omnigent.spec.types import LLMConfig

_logger = logging.getLogger(__name__)


# Env-var keys read by the wrap (see databricks_supervisor_harness.py for the
# contract).
_ENV_MODEL = "HARNESS_SUPERVISOR_MODEL"
_ENV_DATABRICKS_PROFILE = "HARNESS_SUPERVISOR_DATABRICKS_PROFILE"
_ENV_TOOLS_JSON = "HARNESS_SUPERVISOR_TOOLS_JSON"
_ENV_CONNECTION_JSON = "HARNESS_SUPERVISOR_CONNECTION_JSON"


# Runtime status string → inner ToolCallStatus enum. Derived from
# the enum's own ``.value`` so a new status added on either side
# stays in sync. Unknown statuses fall through to ERROR at the call
# site so a silent success-coded surprise can't slip through.
_TOOL_CALL_STATUS_MAP: dict[str, ToolCallStatus] = {s.value: s for s in ToolCallStatus}


# Default per-turn HTTP read timeout (seconds) when not overridden via
# ``config.extra["request_timeout"]``. 300s is the LLMConfig default;
# the runtime executor's own DEFAULT_GATEWAY_READ_TIMEOUT_S (600s)
# only kicks in if llm_config.request_timeout is None.
_DEFAULT_REQUEST_TIMEOUT_S = 300


def _resolve_env_json(env_key: str, expected_type: type) -> Any | None:  # type: ignore[explicit-any]
    """
    Decode a JSON env var, returning ``None`` if unset / empty.

    JSON decode failures and type mismatches RAISE — these env vars
    are written by the parent omnigent process when it spawns the
    supervisor harness subprocess; a malformed payload is parent-side
    misbehavior, not a recoverable user mistake. Silent fallback would
    drop, e.g., an explicit ``base_url`` / ``api_key`` connection
    override and route requests at a different workspace than the
    caller asked for.

    :param env_key: The env-var name, e.g. ``"HARNESS_SUPERVISOR_TOOLS_JSON"``.
    :param expected_type: The type the decoded payload must match
        (``list`` or ``dict``).
    :returns: The parsed value, or ``None`` if the env var is unset
        or empty.
    :raises ValueError: When the env var is set but cannot be decoded
        as JSON, or decodes to the wrong shape.
    """
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{env_key} is set but is not valid JSON ({exc}). This env var "
            f"is written by the parent omnigent spawn path; a malformed "
            f"value is a bug there, not a user-recoverable input."
        ) from exc
    if not isinstance(parsed, expected_type):
        raise ValueError(
            f"{env_key} decoded to {type(parsed).__name__}, expected {expected_type.__name__}."
        )
    return parsed


def _build_supervisor_executor() -> _RuntimeSupervisorExecutor:
    """
    Construct a runtime :class:`SupervisorExecutor` from env vars.

    Called lazily by the inner :class:`SupervisorExecutor` on first
    :meth:`run_turn` so credential resolution failures surface at
    request time, not at FastAPI boot. Resolution order matches the
    runtime executor's own ``from_spec``: explicit
    ``CONNECTION_JSON`` (when both keys populated) wins, else
    profile-based resolution.

    :returns: A configured runtime executor ready for ``run_turn``.
    :raises ValueError: ``HARNESS_SUPERVISOR_MODEL`` unset or empty.
    :raises OSError: Credential resolution failed — bubbles up from
        :func:`resolve_databricks_workspace`.
    """
    model = os.environ.get(_ENV_MODEL, "").strip()
    if not model:
        raise ValueError(
            f"{_ENV_MODEL} is required for the supervisor harness — set "
            f"it to the gateway model identifier (e.g. "
            f"'databricks-claude-sonnet-4-6'). "
            f"The spawn-env builder in workflow.py threads "
            f"spec.llm.model into this var."
        )
    supervisor_tools = _resolve_env_json(_ENV_TOOLS_JSON, list)
    connection = _resolve_env_json(_ENV_CONNECTION_JSON, dict)

    if connection is not None:
        explicit_base = connection.get("base_url")
        explicit_key = connection.get("api_key")
        if explicit_base and explicit_key:
            return _RuntimeSupervisorExecutor(
                model=model,
                supervisor_tools=supervisor_tools,
                base_url=explicit_base.rstrip("/"),
                api_key=explicit_key,
            )

    # Profile-based resolution; resolver fails loud when no source
    # yields creds.
    from omnigent.runtime.credentials.databricks import (
        resolve_databricks_workspace,
    )

    profile = os.environ.get(_ENV_DATABRICKS_PROFILE) or None
    creds = resolve_databricks_workspace(profile)
    from omnigent.inner.databricks_supervisor_gateway import GATEWAY_PATH

    return _RuntimeSupervisorExecutor(
        model=model,
        supervisor_tools=supervisor_tools,
        base_url=f"{creds.host}{GATEWAY_PATH}",
        api_key=creds.token,
    )


def _translate_event(
    event: RuntimeTextChunk
    | RuntimeToolCallObserved
    | RuntimeTurnComplete
    | RuntimeExecutorError
    | object,
) -> list[ExecutorEvent]:
    """
    Translate one runtime event into zero-or-more inner events.

    ToolCallObserved fans out to a paired ToolCallRequest +
    ToolCallComplete (so the adapter can emit them as a
    function_call + function_call_output pair, which Omnigent re-pairs
    back into a single ToolCallObserved); everything else is 1:1.

    The ``object`` arm of the union exists for the warn-and-drop
    fallback when the runtime gains a new event type we haven't
    mapped yet.

    :param event: A runtime event from the wrapped executor.
    :returns: Inner events to yield, in order.
    """
    if isinstance(event, RuntimeTextChunk):
        return [TextChunk(text=event.text)]

    if isinstance(event, RuntimeToolCallObserved):
        metadata: ToolCallMetadata = {"call_id": event.call_id}
        status = _TOOL_CALL_STATUS_MAP.get(event.status, ToolCallStatus.ERROR)
        return [
            ToolCallRequest(name=event.name, args=event.arguments, metadata=metadata),
            ToolCallComplete(
                name=event.name,
                status=status,
                result=event.result,
                # Runtime ToolCallObserved has no error-message field,
                # only a status string.
                error=None,
                duration_ms=event.duration_ms,
                metadata=metadata,
            ),
        ]

    if isinstance(event, RuntimeTurnComplete):
        # Runtime-only fields (usage, response_model, response_id,
        # finish_reasons) are dropped — see Known limitations.
        return [TurnComplete(response=event.text)]

    if isinstance(event, RuntimeExecutorError):
        # Fold the runtime ``code`` into the message prefix; inner
        # ExecutorError has no separate code field.
        prefix = f"[{event.code}] " if getattr(event, "code", None) else ""
        return [ExecutorError(message=f"{prefix}{event.message}", retryable=event.retryable)]

    _logger.warning(
        "Inner SupervisorExecutor: dropping unknown runtime event type %s",
        type(event).__name__,
    )
    return []


# Module-level stub :class:`ExecutorContext` for the runtime
# ``run_turn`` call. The runtime supervisor doesn't use the context
# (gateway runs every tool server-side), but the ABC signature
# requires one. The callbacks raise if invoked so a future contract
# change surfaces loud rather than running a no-op. Singleton because
# the context is stateless and immutable — no reason to rebuild per
# turn.


async def _refuse_call_tool(_: RuntimeToolCallRequested) -> NoReturn:
    raise RuntimeError("supervisor should never invoke context.call_tool")


async def _refuse_enforce(_name: str, _args: ToolArgs) -> NoReturn:
    raise RuntimeError("supervisor should never invoke context.enforce_tool_call_policy")


_THROWAWAY_CONTEXT = ExecutorContext(
    task_id="",
    conversation_id="",
    storage_dir=Path("/tmp"),
    call_tool=_refuse_call_tool,
    enforce_tool_call_policy=_refuse_enforce,
)


class SupervisorExecutor(Executor):
    """
    Inner-side wrapper around the runtime
    :class:`omnigent.inner.databricks_supervisor_gateway.SupervisorExecutor`.

    Delegates each turn to a lazily-constructed runtime executor and
    translates its events to the inner vocabulary. Lazy construction
    surfaces credential-resolution failures at request time, not at
    FastAPI boot.
    """

    def __init__(self) -> None:
        """Construct an idle wrapper; runtime executor is built on first use."""
        self._runtime: _RuntimeSupervisorExecutor | None = None

    def _runtime_executor(self) -> _RuntimeSupervisorExecutor:
        """Return the cached runtime executor, building it on first call."""
        if self._runtime is None:
            self._runtime = _build_supervisor_executor()
        return self._runtime

    async def run_turn(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[ExecutorEvent]:
        """
        Run one supervisor turn — delegate to the runtime executor and
        translate events to the inner vocabulary.

        :param messages: Conversation history (Responses-API input
            items); forwarded verbatim.
        :param tools: Ignored — supervisor tools come from
            ``HARNESS_SUPERVISOR_TOOLS_JSON``, not workflow.
        :param system_prompt: Forwarded as the request's
            ``instructions`` field.
        :param config: Only ``config.extra["request_timeout"]`` is
            honored (overrides the HTTP read deadline); other fields
            are gateway-controlled.
        """
        del tools
        runtime_executor = self._runtime_executor()
        request_timeout = _DEFAULT_REQUEST_TIMEOUT_S
        if config is not None and config.extra:
            extra_timeout = config.extra.get("request_timeout")
            if isinstance(extra_timeout, int) and extra_timeout > 0:
                request_timeout = extra_timeout
        extra: dict[str, Any] = {}
        if config is not None and config.extra.get("reasoning_effort"):
            extra["reasoning_effort"] = config.extra["reasoning_effort"]
        llm_config = LLMConfig(
            model=runtime_executor.model,
            request_timeout=request_timeout,
            extra=extra,
        )
        async for runtime_event in runtime_executor.run_turn(
            messages=messages,
            tools=[],
            system_prompt=system_prompt,
            llm_config=llm_config,
            context=_THROWAWAY_CONTEXT,
        ):
            for inner_event in _translate_event(runtime_event):
                yield inner_event

    def supports_streaming(self) -> bool:
        """:returns: ``True`` — gateway streams SSE events."""
        return True

    def supports_tool_calling(self) -> bool:
        """:returns: ``True`` — tools execute server-side."""
        return True

    def handles_tools_internally(self) -> bool:
        """:returns: ``True`` — gateway already ran each tool; session must not re-execute."""
        return True

    def max_context_tokens(self) -> int | None:
        """:returns: ``None`` — gateway manages its own context window."""
        return None

    async def close_session(self, session_key: str) -> None:
        """No per-session state to release; runtime executor builds a fresh httpx client per turn.

        :param session_key: Unused.
        """
        del session_key

    async def interrupt_session(self, session_key: str) -> bool:
        """:returns: Always ``False`` — gateway has no mid-stream cancellation surface.

        :param session_key: Unused.
        """
        del session_key
        return False

    async def enqueue_session_message(self, session_key: str, content: EnqueuedContent) -> bool:
        """:returns: Always ``False`` — supervisor has no mid-turn message queue.

        :param session_key: Unused.
        :param content: Unused.
        """
        del session_key, content
        return False

    async def close(self) -> None:
        """No-op — runtime executor opens an httpx client per turn under ``async with``."""
