"""SupervisorExecutor: server-runs-tools executor for the Databricks
``ai-gateway/mlflow/v1/responses`` endpoint.

Delegates the entire turn — model call AND tool execution — to the
gateway. Translates the gateway's OpenAI Responses-style SSE stream
into omnigent :class:`ExecutorEvent` instances; emits
:class:`ToolCallObserved` directly (no preceding
:class:`ToolCallRequested`) because the gateway runs each tool
internally.

For the architectural overview, SSE event vocabulary, OAuth flow,
known limitations (no interruption / no background mode / no client-
side tools / TurnComplete metadata drop / etc.), and the recipe for
adding a new gateway tool type, see
``designs/DATABRICKS_SUPERVISOR_API_INTEGRATION.md``. This module is the
code-level reference; the design doc is the integration-level one.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass, field
from typing import Any

import httpx
from httpx_sse import aconnect_sse
from typing_extensions import Self

from omnigent.reasoning_effort import OPENAI_EFFORTS, validate_effort
from omnigent.runtime.credentials.databricks import (
    WorkspaceCreds,
    resolve_databricks_workspace,
)
from omnigent.runtime.executors.base import (
    Executor,
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    TextChunk,
    ToolCallObserved,
    TurnComplete,
)
from omnigent.spec import AgentSpec
from omnigent.spec.types import LLMConfig

_logger = logging.getLogger(__name__)

# ── Module-level constants ─────────────────────────────────────

# Path appended to the resolved Databricks workspace host to form
# the supervisor gateway base URL. Provided literally because there
# is exactly one supported gateway endpoint shape today; widening
# this to a configurable URL would be a spec-self-containment
# antipattern (see CLAUDE.md design principle 1).
GATEWAY_PATH: str = "/ai-gateway/mlflow/v1"

# Path appended to the gateway base URL for streaming responses.
RESPONSES_PATH: str = "/responses"

# Default per-request timeout for the gateway HTTP call. The
# gateway streams SSE events as the supervisor runs tools, so the
# overall connection may be open for several minutes; pick a
# generous timeout that still bounds runaway requests. The pool
# and connect timeouts default to httpx's library defaults via
# ``httpx.Timeout(read=...)``-style construction below.
DEFAULT_GATEWAY_READ_TIMEOUT_S: float = 600.0
DEFAULT_GATEWAY_CONNECT_TIMEOUT_S: float = 30.0

# Regex for parsing the OAuth-required error message emitted by the
# Databricks supervisor when a tool call needs user login. The real
# message format (captured from staging) reads:
#
#   "...Credential for user identity('NNN') is not found for the
#    connection 'system_ai_agent_google_drive'. Please login first
#    to the connection by visiting <URL>"
#
# The connector name is in single quotes after ``connection`` and
# the login URL is at the end (after ``visiting`` or ``at``,
# preceded by whitespace).
_OAUTH_LOGIN_RE: re.Pattern[str] = re.compile(
    r"connection\s+'(?P<connector>[^']+)'.*?(?:visiting|at)\s+(?P<url>https?://\S+)",
    re.DOTALL,
)


@dataclass(frozen=True)
class GatewayEndpoint:
    """
    Resolved Databricks supervisor-gateway endpoint.

    Returned by :func:`_resolve_gateway_credentials` so the call
    site has named fields rather than positionally-fragile tuple
    unpacking (CLAUDE.md rule 18).

    :param base_url: The full endpoint base, with no trailing slash,
        already including ``/ai-gateway/mlflow/v1`` so callers append
        only ``/responses``.
    :param api_key: The bearer token to inject as
        ``Authorization: Bearer <api_key>``.
    """

    base_url: str
    api_key: str


def _resolve_gateway_credentials(spec: AgentSpec) -> GatewayEndpoint:
    """
    Resolve the supervisor-gateway endpoint (base URL + bearer) from a spec.

    Resolution order:

    1. If ``spec.executor.connection`` provides BOTH ``base_url`` and
       ``api_key``, use them verbatim. The supplied ``base_url`` is
       expected to be the FULL endpoint including ``/ai-gateway/mlflow/v1``
       (we do not append the gateway path; the user has full control).
    2. If exactly one of ``base_url`` / ``api_key`` is set,
       :class:`ValueError` — partial overrides are almost always an
       authoring mistake (the caller intended one workspace but the
       resolver would silently fall back to ambient creds, sending
       requests to a different workspace and confusing debugging).
    3. Otherwise call :func:`resolve_databricks_workspace` with
       ``spec.executor.profile`` and compose ``{host}{GATEWAY_PATH}``.

    :param spec: The agent spec; ``spec.executor.model`` must already
        be non-None (the caller has checked).
    :returns: A :class:`GatewayEndpoint` carrying base URL + token.
    :raises ValueError: When ``spec.executor.connection`` is partially
        populated (exactly one of ``base_url``/``api_key`` set).
    """
    connection = spec.executor.connection or {}
    explicit_base = connection.get("base_url")
    explicit_key = connection.get("api_key")
    if explicit_base and explicit_key:
        # The user supplied both — trust them verbatim. Strip a
        # trailing slash on base_url so URL composition stays clean.
        return GatewayEndpoint(base_url=explicit_base.rstrip("/"), api_key=explicit_key)
    if bool(explicit_base) != bool(explicit_key):
        # Exactly one populated; treat as a spec error rather than
        # silently falling back to ambient credentials (which would
        # send requests to a different workspace than the user meant).
        missing = "api_key" if explicit_base else "base_url"
        present = "base_url" if explicit_base else "api_key"
        raise ValueError(
            f"spec.executor.connection populates {present!r} but not {missing!r}; "
            "supervisor requires both keys together (or neither, in which "
            "case credentials are resolved from the profile / ~/.databrickscfg)."
        )
    creds: WorkspaceCreds = resolve_databricks_workspace(spec.executor.profile)
    # ``creds.host`` already has trailing slashes stripped by the
    # resolver, so simple concatenation produces a valid URL.
    return GatewayEndpoint(base_url=f"{creds.host}{GATEWAY_PATH}", api_key=creds.token)


class BearerAuth(httpx.Auth):
    """
    httpx auth flow that injects ``Authorization: Bearer <token>``.

    A minimal inline implementation so the supervisor executor can
    talk to the Databricks gateway without pulling in
    ``databricks-ai-bridge`` as a dependency yet.

    :param token: The bearer token from
        :class:`WorkspaceCreds.token`.
    """

    # TODO: switch to databricks_openai.utils.clients._resolve_base_url /
    # BearerAuth once databricks-ai-bridge is a dep, so we inherit
    # upstream fixes for free.
    def __init__(self, token: str) -> None:
        """
        Store the bearer token.

        :param token: The bearer token from
            :class:`WorkspaceCreds.token`.
        """
        self._token = token

    def auth_flow(
        self,
        request: httpx.Request,
    ) -> Generator[httpx.Request, httpx.Response, None]:
        """
        Inject the Authorization header and yield the modified request.

        httpx's :class:`httpx.Auth` API is synchronous-generator-based
        regardless of whether the underlying client is sync or async,
        so this method is a plain ``Generator``, not ``AsyncIterator``.

        :param request: The outgoing httpx request.
        :returns: A generator yielding the request with the
            Authorization header populated.
        """
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


class SupervisorExecutor(Executor):
    """
    Agent-plane executor backed by the Databricks supervisor gateway.

    Constructed by :meth:`from_spec` given an :class:`AgentSpec` with
    the ``databricks_supervisor`` harness. Each :meth:`run_turn` call POSTs
    one streaming request to ``{base_url}{RESPONSES_PATH}`` and
    translates the streamed events into omnigent
    :class:`ExecutorEvent` instances.

    :param model: The model identifier from
        :attr:`AgentSpec.llm.model`, e.g.
        ``"databricks-claude-sonnet-4-6"``.
    :param supervisor_tools: The verbatim typed tool declarations
        from :attr:`ExecutorSpec.supervisor_tools`, e.g.
        ``[{"type": "uc_connection",
        "uc_connection": {"name": "...", "description": "..."}}]``.
        ``None`` when no tools are declared (the supervisor still
        works as a plain LLM in that case).
    :param base_url: The fully-resolved gateway base URL, e.g.
        ``"https://example.databricks.com/ai-gateway/mlflow/v1"``.
    :param api_key: The bearer token for the gateway.
    :param http_client: Optional pre-built :class:`httpx.AsyncClient`.
        Tests inject a stub client; production code passes ``None``
        and lets :meth:`run_turn` build a fresh client per call.
    """

    def __init__(
        self,
        *,
        model: str,
        supervisor_tools: list[dict[str, Any]] | None,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """
        Store the gateway parameters; callers should prefer
        :meth:`from_spec`.

        :param model: The model identifier from the agent spec.
        :param supervisor_tools: The typed (nested) tool list.
            ``Any`` for the value type because the per-type config
            sub-dict (e.g. ``{"name": "...", "description": "..."}``
            or ``{"id": "..."}``) carries heterogeneous fields and
            round-trips verbatim to the gateway.
        :param base_url: The fully-resolved gateway base URL.
        :param api_key: The bearer token for the gateway.
        :param http_client: Optional pre-built async client (for
            tests). Production callers pass ``None``.
        """
        self._model = model
        self._supervisor_tools = supervisor_tools
        self._base_url = base_url
        self._api_key = api_key
        self._http_client = http_client

    @property
    def model(self) -> str:
        """:returns: The configured gateway model identifier."""
        return self._model

    @classmethod
    def from_spec(
        cls,
        spec: AgentSpec,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> Self:
        """
        Build a :class:`SupervisorExecutor` from an :class:`AgentSpec`.

        Resolution order for the gateway URL + auth:

        1. If ``spec.executor.connection`` provides BOTH ``base_url`` and
           ``api_key``, use them verbatim.
        2. Otherwise, call
           :func:`resolve_databricks_workspace` with
           ``spec.executor.profile`` and compose
           ``{host}{GATEWAY_PATH}`` (the host has no trailing slash
           by contract).

        :param spec: The agent spec. Must use the
            ``databricks_supervisor`` harness and have a non-None
            ``executor.model`` set.
        :param http_client: Optional pre-built async client. Tests
            inject one (with a :class:`httpx.MockTransport`) so they
            can verify the eventual request URL through the public
            entry point. Production callers leave it ``None`` and
            :meth:`run_turn` builds a fresh client per call.
        :returns: A configured executor instance.
        :raises ValueError: If ``spec.executor.model`` is missing —
            the gateway requires a model identifier and we fail loud
            rather than substitute a default.
        """
        if not spec.executor.model:
            raise ValueError(
                "SupervisorExecutor.from_spec requires spec.executor.model "
                "to be set; got " + repr(spec.executor.model)
            )
        endpoint = _resolve_gateway_credentials(spec)
        return cls(
            model=spec.executor.model,
            # ``supervisor_tools`` round-trips verbatim from the
            # parser into the gateway request body — the executor
            # does not reshape the nested ``{type, <type>: {...}}``
            # entries.
            supervisor_tools=spec.executor.supervisor_tools,
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            http_client=http_client,
        )

    def max_context_tokens(self) -> int | None:
        """
        Return ``None`` so the workflow skips compaction.

        :returns: Always ``None``.
        """
        # executor owns compaction; the supervisor's server-side loop is
        # opaque, so the workflow must skip its @step wrap and never
        # inject compaction.
        return None

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],  # noqa: ARG002
        system_prompt: str,
        llm_config: LLMConfig,
        context: ExecutorContext,  # noqa: ARG002
    ) -> AsyncIterator[ExecutorEvent]:
        """
        Run one supervisor turn and yield translated events.

        Sends a streaming POST to ``{base_url}{RESPONSES_PATH}`` and
        translates the SSE stream into omnigent events via
        :func:`_translate_event`. asyncio.CancelledError propagates
        between events but the in-flight gateway step (one HTTP read)
        completes first — the gateway has no mid-step cancellation
        point.

        :param messages: Conversation history as Responses API input
            items. Forwarded verbatim as the request ``input`` field.
        :param tools: Workflow-supplied tool schemas; ignored. The
            supervisor uses :attr:`_supervisor_tools` (typed
            declarations from the spec) instead — its tools live
            on the gateway side, not in Omnigent' tool manager.
        :param system_prompt: System instructions; forwarded as the
            request ``instructions`` field. Empty string omits the
            field entirely (the gateway treats absent and empty
            equivalently, but omitting is cleaner on the wire).
        :param llm_config: LLM configuration; ``request_timeout``
            governs the HTTP read timeout.
        :param context: Workflow capabilities; unused — the
            supervisor runs every tool itself.
        """
        body = self._build_request_body(messages, system_prompt, llm_config)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        # Honor llm.request_timeout for the read deadline (matches the
        # docstring contract). Connect/pool stay at the conservative
        # default since a slow workspace endpoint usually means
        # network/auth trouble worth surfacing fast.
        read_timeout = (
            float(llm_config.request_timeout)
            if llm_config.request_timeout
            else DEFAULT_GATEWAY_READ_TIMEOUT_S
        )
        timeout = httpx.Timeout(
            connect=DEFAULT_GATEWAY_CONNECT_TIMEOUT_S,
            read=read_timeout,
            write=read_timeout,
            pool=DEFAULT_GATEWAY_CONNECT_TIMEOUT_S,
        )
        url = f"{self._base_url}{RESPONSES_PATH}"
        # Allow tests to inject a stub client (with pre-built SSE
        # streams). Production code constructs a fresh client per
        # turn so connection state is never reused across turns.
        if self._http_client is not None:
            async for event in _stream_with_client(self._http_client, url, body, headers):
                yield event
            return
        async with httpx.AsyncClient(
            auth=BearerAuth(self._api_key),
            timeout=timeout,
        ) as client:
            async for event in _stream_with_client(client, url, body, headers):
                yield event

    def _build_request_body(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        llm_config: LLMConfig | None = None,
    ) -> dict[str, Any]:
        """
        Build the JSON body for the gateway streaming request.

        :param messages: Conversation history items.
        :param system_prompt: System instructions; an empty string
            is treated as "no instructions" and the field is
            omitted from the body.
        :returns: A JSON-serializable dict ready to POST.
        """
        body: dict[str, Any] = {
            "model": self._model,
            "input": messages,
            # ``supervisor_tools`` round-trips verbatim — the gateway
            # is the consumer of these typed declarations, not us.
            "tools": self._supervisor_tools or [],
            "stream": True,
        }
        if system_prompt:
            body["instructions"] = system_prompt
        reasoning_effort = None
        if llm_config is not None:
            reasoning_effort = llm_config.extra.get("reasoning_effort")
        if reasoning_effort:
            effort = validate_effort(
                reasoning_effort, "Databricks supervisor gateway", OPENAI_EFFORTS
            )
            body["reasoning"] = {"effort": effort}
        return body


@dataclass
class _StreamState:
    """
    Mutable cross-event state for one supervisor SSE stream.

    Carried through the per-event handler so the orchestrator
    (:func:`_stream_with_client`) doesn't have to thread half-a-
    dozen variables manually. Every field starts at the
    "no events seen yet" value and is updated by
    :func:`_handle_one_event` as the stream advances.

    :param saw_completion: Set ``True`` when a real
        :class:`TurnComplete` event has been yielded (NOT when the
        lying-terminator path swallows one).
    :param saw_error_event: Set ``True`` after any ``error`` SSE
        event fires. Used to suppress the gateway's lying
        ``response.completed`` terminator.
    :param streamed_text_parts: Accumulator of every
        ``response.output_text.delta`` value, in order. Re-joined
        as the fallback ``TurnComplete.text`` when the terminator
        carries an empty ``response.output``.
    :param pending_oauth_text: The auth-required TextChunk's text
        when an OAuth ``error`` event has fired. Surfaces as the
        synthesized terminal ``TurnComplete.text`` so the workflow
        persists the actionable login URL instead of an empty
        assistant message.
    :param pending_tool_calls: Buffer of ``function_call`` items
        keyed by ``call_id``. Cleared as their matching
        ``function_call_output`` events arrive; any remaining
        entries flush at stream end with ``result=""``.
    """

    saw_completion: bool = False
    saw_error_event: bool = False
    streamed_text_parts: list[str] = field(default_factory=list)
    pending_oauth_text: str | None = None
    pending_tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)


async def _stream_with_client(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> AsyncIterator[ExecutorEvent]:
    """
    Issue the streaming POST and yield translated events.

    Orchestrator: opens the SSE stream, dispatches each event to
    :func:`_handle_one_event` (which mutates the shared
    :class:`_StreamState`), then flushes any state left over via
    :func:`_flush_stream_state`. HTTP errors short-circuit through
    :func:`_emit_http_error`. Gateway-quirk handling lives in the
    handler functions; see :class:`_StreamState` and the module
    docstring for an inventory.

    :param client: The httpx async client to use.
    :param url: The full gateway URL (base_url + RESPONSES_PATH).
    :param body: The JSON body for the POST.
    :param headers: Extra request headers (Content-Type, Accept).
    :returns: An async iterator of omnigent executor events.
    """
    # The supervisor's gateway-side step is opaque — once we open the
    # SSE stream we cannot cancel a tool mid-execution. asyncio.
    # CancelledError raised between events propagates through the
    # ``async for`` loop and tears the connection down at the next
    # ``aiter_sse`` boundary; the in-flight tool finishes first.
    state = _StreamState()
    try:
        async with aconnect_sse(client, "POST", url, json=body, headers=headers) as event_source:
            response = event_source.response
            if response.status_code >= 400:
                async for event in _emit_http_error(response):
                    yield event
                return
            async for sse in event_source.aiter_sse():
                payload = _parse_sse_data(sse.data)
                if payload is None:
                    continue
                # The gateway's ``error`` events carry the type
                # on the SSE ``event:`` line, not as a ``type``
                # field in the JSON data. Fall back to ``sse.event``
                # so downstream handlers see a consistent shape.
                if "type" not in payload and sse.event:
                    payload["type"] = sse.event
                for event in _handle_one_event(payload, state):
                    yield event
    except httpx.HTTPError as exc:
        # Flush any state observed before the error tore the stream
        # down — otherwise accumulated text deltas and tool-call
        # observations are silently dropped (a read-timeout 9 minutes
        # into a turn would lose every observed tool call).
        for event in _flush_stream_state(state):
            yield event
        # Only mark connection-class failures as retryable. Mid-stream
        # protocol errors (e.g. ``RemoteProtocolError``) are NOT safe
        # to retry: server-side tools may have already produced
        # observable side effects that re-running would duplicate.
        retryable = isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
        yield ExecutorError(
            message=f"Supervisor gateway HTTP error: {exc}",
            code="http_error",
            retryable=retryable,
        )
        return
    for event in _flush_stream_state(state):
        yield event


def _handle_one_event(
    payload: dict[str, Any], state: _StreamState
) -> Generator[ExecutorEvent, None, None]:
    """
    Translate one parsed SSE payload into omnigent events,
    mutating *state* as needed for cross-event quirks.

    Three special-case branches before the generic dispatch:

    1. Accumulate ``response.output_text.delta`` values into
       ``state.streamed_text_parts`` so the terminator can fall
       back to them when ``response.output`` is empty.
    2. Suppress the gateway's lying ``response.completed`` after
       an ``error`` event already fired (do NOT mark
       ``saw_completion`` — the post-stream synthesizer below
       still needs to fire to carry the OAuth message forward).
    3. Pair ``response.output_item.done`` with
       ``state.pending_tool_calls`` via
       :func:`_translate_output_item_done` (function_call ↔
       function_call_output correlation).

    Every other event type runs through the pure
    :func:`_translate_event` and gets dispatched normally; OAuth
    errors additionally stash their message text on the state so
    :func:`_flush_stream_state` can synthesize a terminal
    :class:`TurnComplete` that carries it.

    :param payload: The parsed SSE event payload.
    :param state: Cross-event state mutated in-place as a side
        effect; never returned.
    :yields: The omnigent events derived from this one
        payload, in order.
    """
    event_type = payload.get("type")

    if event_type == "response.output_text.delta":
        delta = payload.get("delta")
        if isinstance(delta, str):
            state.streamed_text_parts.append(delta)

    if state.saw_error_event and event_type == "response.completed":
        # Lying-terminator suppression. Intentionally do NOT mark
        # saw_completion — _flush_stream_state needs to fire so
        # the OAuth auth message (or streamed text) survives.
        return

    if event_type == "response.output_item.done":
        observed = _translate_output_item_done(payload, state.pending_tool_calls)
        if observed is not None:
            yield observed
        return

    # Only the response.completed translator consumes streamed_text
    # (as a fallback when the terminator's ``response.output`` is
    # empty). Skip the join for every other event type.
    streamed_text = (
        _join(state.streamed_text_parts) if event_type == "response.completed" else None
    )
    for translated in _translate_event(payload, streamed_text=streamed_text):
        if event_type == "error" and isinstance(translated, ExecutorError | TextChunk):
            state.saw_error_event = True
            if isinstance(translated, TextChunk) and payload.get("code") == "oauth":
                # Stash the auth message; the post-stream
                # synthesizer surfaces it as a TurnComplete so the
                # workflow has a terminal event to persist.
                state.pending_oauth_text = translated.text
        if isinstance(translated, TurnComplete):
            state.saw_completion = True
        yield translated


def _flush_stream_state(
    state: _StreamState,
) -> Generator[ExecutorEvent, None, None]:
    """
    Emit any events that the per-event handler couldn't surface
    until the stream closed.

    Two cases:

    1. **Unpaired tool invocations.** Any ``function_call`` whose
       ``function_call_output`` never arrived (OAuth-required
       turn, gateway hiccup, cancellation) flushes here with
       ``result=""`` so the invocation is at least visible in
       history.
    2. **No real terminator.** When the loop ended without a
       :class:`TurnComplete` (either we suppressed a lying
       ``response.completed`` after an error, or the stream
       simply truncated), synthesize one. Prefer the OAuth auth
       text when present (so reconnect/history shows the
       actionable login URL); otherwise carry whatever streamed
       text we accumulated.

    :param state: The fully-populated :class:`_StreamState` after
        the SSE loop has exited.
    :yields: Zero or more terminal events.
    """
    for item in state.pending_tool_calls.values():
        flushed = _build_tool_call_observed(item, result_text="")
        if flushed is not None:
            yield flushed
    state.pending_tool_calls.clear()
    if not state.saw_completion:
        yield TurnComplete(
            text=state.pending_oauth_text or _join(state.streamed_text_parts),
            usage=None,
            response_model=None,
            response_id=None,
            finish_reasons=None,
        )


def _join(parts: list[str]) -> str | None:
    """
    Join accumulated text deltas, returning ``None`` for an empty list.

    :param parts: The list of text deltas captured during streaming.
    :returns: The concatenated text, or ``None`` when no deltas
        were captured (so :class:`TurnComplete` reports
        "no text" rather than an empty string).
    """
    if not parts:
        return None
    return "".join(parts)


async def _emit_http_error(
    response: httpx.Response,
) -> AsyncIterator[ExecutorEvent]:
    """
    Read a non-2xx response body and yield a single
    :class:`ExecutorError`.

    Extracted to keep :func:`_stream_with_client` under the 40-line
    function-length budget and to isolate the body-decode error
    handling.

    :param response: An httpx response whose ``status_code`` is
        already known to be 4xx or 5xx.
    :returns: An async iterator yielding exactly one
        :class:`ExecutorError`.
    """
    try:
        error_body = await response.aread()
        error_text = error_body.decode("utf-8", errors="replace")
    except (httpx.HTTPError, UnicodeError) as exc:
        error_text = f"<error reading body: {exc!r}>"
    yield ExecutorError(
        message=(f"Supervisor gateway returned HTTP {response.status_code}: {error_text}"),
        code=f"http_{response.status_code}",
        # 5xx is a server transient — retryable so the surrounding
        # retry policy can reissue. 4xx is a client/spec error.
        retryable=response.status_code >= 500,
    )


def _parse_sse_data(raw: str) -> dict[str, Any] | None:
    """
    Parse one SSE ``data:`` payload as JSON.

    Returns ``None`` for non-JSON payloads (heartbeats, ``[DONE]``
    markers, blank lines) so callers can simply ``continue``.

    :param raw: The raw ``data`` string from one SSE event.
    :returns: The parsed JSON dict, or ``None`` when the payload is
        not a JSON object.
    """
    if not raw or raw == "[DONE]":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Non-JSON in-stream payloads almost always signal a bad
        # intermediate proxy frame (e.g. an HTML 502 splat from a
        # gateway). Without a log, we'd silently treat it as a
        # heartbeat and the caller would wait forever for a real
        # event. Truncate aggressively — these can be large.
        _logger.warning(
            "supervisor SSE stream produced a non-JSON data payload; dropping. raw[:200]=%r",
            raw[:200],
        )
        return None
    if not isinstance(parsed, dict):
        _logger.warning(
            "supervisor SSE stream produced a non-object JSON payload; dropping. parsed=%r",
            parsed,
        )
        return None
    return parsed


def _translate_event(
    payload: dict[str, Any],
    *,
    streamed_text: str | None = None,
) -> list[ExecutorEvent]:
    """
    Translate one supervisor SSE event payload into omnigent events.

    Pure / stateless. ``response.output_item.done`` events are NOT
    translated here — they require cross-event state (pairing
    ``function_call`` items with their ``function_call_output``
    siblings) and are handled by
    :func:`_translate_output_item_done` which is called directly
    from :func:`_stream_with_client`.

    Recognized event shapes (per the captured staging fixture):

    - ``response.created``, ``response.in_progress``,
      ``response.output_item.added``,
      ``response.function_call_arguments.done`` → no-op.
    - ``response.output_text.delta`` → :class:`TextChunk`.
    - ``error`` with ``code == "oauth"`` → :class:`TextChunk` with
      login instructions.
    - ``error`` with any other code → :class:`ExecutorError`.
    - ``response.completed`` → :class:`TurnComplete`.

    :param payload: The parsed JSON payload from one SSE event.
    :param streamed_text: Concatenation of every
        ``response.output_text.delta`` seen so far on this stream.
        Used as a fallback when the terminal ``response.completed``
        carries an empty ``response.output`` even though deltas
        streamed earlier (a real gateway shape captured in the
        fixture).
    :returns: A list of zero-or-more omnigent events.
    """
    event_type = payload.get("type")
    if event_type == "response.output_text.delta":
        delta = payload.get("delta")
        if isinstance(delta, str) and delta:
            return [TextChunk(text=delta)]
        return []
    if event_type == "error":
        return [_translate_error_event(payload)]
    if event_type == "response.completed":
        return [_translate_response_completed(payload, streamed_text=streamed_text)]
    # response.created, response.in_progress,
    # response.output_item.added,
    # response.function_call_arguments.done — all consumed silently.
    # response.output_item.done — handled in _stream_with_client.
    return []


def _translate_output_item_done(
    payload: dict[str, Any],
    pending: dict[str, dict[str, Any]],
) -> ToolCallObserved | None:
    """
    Pair a ``response.output_item.done`` event with the buffer of
    in-flight function calls and emit at most one :class:`ToolCallObserved`.

    Two item shapes matter on this event:

    - ``item.type == "function_call"`` carries the model's
      invocation (call_id, name, JSON-stringified arguments). We
      BUFFER the item by ``call_id`` and emit nothing — the result
      event is expected to follow.
    - ``item.type == "function_call_output"`` carries the
      server-executed tool's result (call_id, output). We look up
      the buffered invocation by ``call_id``, build a single
      :class:`ToolCallObserved` carrying both, and remove the
      pending entry.

    Other item types (assistant ``message`` items, etc.) are
    consumed silently — their text already streamed via
    ``response.output_text.delta`` events and would double-render
    if surfaced again here.

    :param payload: The SSE event payload, expected to carry an
        ``item`` mapping.
    :param pending: The cross-event buffer keyed by ``call_id``.
        This function MUTATES the buffer (stores function_call
        items, deletes them on output).
    :returns: A populated :class:`ToolCallObserved` when a
        ``function_call_output`` matched a buffered invocation;
        ``None`` otherwise (buffered, mismatched, or non-tool item).
    :raises ValueError: When a ``function_call`` item is missing
        BOTH ``call_id`` and ``id`` (correlation impossible).
    """
    item = payload.get("item")
    if not isinstance(item, dict):
        return None
    item_type = item.get("type")
    if item_type == "function_call":
        # Buffer until the matching function_call_output arrives. We
        # validate call_id eagerly so a missing-id fault surfaces at
        # the invocation event, not at flush-time.
        call_id = _resolve_tool_call_id(item)
        pending[call_id] = item
        return None
    if item_type == "function_call_output":
        # Pair with the previously-buffered invocation. The output
        # item carries ``call_id`` (matching the function_call's
        # ``call_id``) and an ``output`` field with the result.
        call_id_raw = item.get("call_id") or item.get("id")
        if not isinstance(call_id_raw, str) or not call_id_raw:
            # Output without a usable correlation key — surface as
            # ToolCallObserved on its own with a synthetic name so
            # the user can see the result text but knows it
            # couldn't be correlated. We still pull out whatever
            # ``output`` text the gateway emitted.
            return _build_uncorrelated_output_observed(item)
        invocation = pending.pop(call_id_raw, None)
        if invocation is None:
            # Output arrived without a matching invocation. Treat
            # the same way as the uncorrelated case so the result
            # is at least visible in the stream.
            return _build_uncorrelated_output_observed(item)
        return _build_tool_call_observed(invocation, result_text=_extract_output_text(item))
    # Other item types (e.g. ``message``) are consumed silently —
    # their text already streamed via output_text.delta events.
    return None


def _extract_output_text(item: dict[str, Any]) -> str:
    """
    Pull the result text from a ``function_call_output`` item.

    The captured fixture isn't yet populated in our test corpus
    (the OAuth-required probe failed before the gateway emitted a
    success output). The Supervisor API documents the field as
    ``output`` carrying either a string (textual result) or a
    structured object (JSON-encoded by us so downstream consumers
    receive a single string).

    :param item: The ``function_call_output`` item.
    :returns: The result as a string. Empty string if no output
        field is present.
    """
    raw = item.get("output")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, default=str)


def _build_uncorrelated_output_observed(
    item: dict[str, Any],
) -> ToolCallObserved:
    """
    Build a :class:`ToolCallObserved` for a ``function_call_output``
    that has no matching buffered invocation.

    Edge case — happens if the gateway emits an output event whose
    ``call_id`` we never saw on a prior ``function_call`` item, or
    whose ``call_id`` is missing entirely. We surface the result
    so it isn't lost; downstream consumers can see it but won't
    have the original tool name or arguments.

    :param item: The ``function_call_output`` item.
    :returns: A :class:`ToolCallObserved` with synthetic name and
        empty arguments, carrying the result text.
    """
    call_id_raw = item.get("call_id") or item.get("id")
    call_id = call_id_raw if isinstance(call_id_raw, str) and call_id_raw else "unknown"
    return ToolCallObserved(
        call_id=call_id,
        # Synthetic name surfaces the issue without crashing the
        # stream — operators reading logs can grep for this string.
        name="<uncorrelated_output>",
        arguments={},
        result=_extract_output_text(item),
        status="success",
        duration_ms=0.0,
    )


def _build_tool_call_observed(
    item: dict[str, Any],
    *,
    result_text: str,
) -> ToolCallObserved | None:
    """
    Build a :class:`ToolCallObserved` from a buffered
    ``function_call`` item plus the corresponding output text.

    :param item: The buffered ``function_call`` item.
    :param result_text: The text from the matching
        ``function_call_output`` item, or ``""`` when the stream
        ended before output arrived.
    :returns: A populated :class:`ToolCallObserved`, or ``None``
        when the item is missing the ``name`` field (malformed).
    """
    name = item.get("name")
    if not isinstance(name, str) or not name:
        return None
    return ToolCallObserved(
        call_id=_resolve_tool_call_id(item),
        name=name,
        arguments=_parse_tool_arguments(item),
        # Prefer the result_text passed in (from a real
        # function_call_output sibling event) over any inline
        # ``result`` field on the function_call item itself. The
        # captured fixture never populates inline results; if a
        # future variant does, the inline value still surfaces via
        # _extract_tool_result when result_text is empty.
        result=result_text or _extract_tool_result(item),
        status=_map_tool_status(item),
        # The gateway does not currently report per-tool wall-clock
        # duration; record 0.0 so downstream consumers can
        # distinguish "no measurement" from a real zero-second tool.
        duration_ms=0.0,
    )


def _translate_error_event(payload: dict[str, Any]) -> ExecutorEvent:
    """
    Translate a top-level streaming ``error`` event.

    Streaming variant shape (top-level ``code``, NOT nested under
    ``error.code``)::

        {"type": "error", "code": "oauth", "message": "...", ...}

    OAuth errors fan out to a user-facing :class:`TextChunk`; every
    other code becomes a non-retryable :class:`ExecutorError`.

    :param payload: The parsed SSE event payload.
    :returns: A :class:`TextChunk` for OAuth, an
        :class:`ExecutorError` for everything else.
    """
    # The gateway uses ``code`` in OAuth errors and ``error_code``
    # in INVALID_PARAMETER_VALUE errors. Accept both.
    code = payload.get("code") or payload.get("error_code")
    # Default to empty string rather than failing loud here because
    # the workflow's error handling renders ``f"Supervisor error
    # ({code}): {message}"`` even with no message — surfacing the
    # bare code is more useful to the user than a parse-time crash
    # if the gateway ever omits the field.
    message = payload.get("message", "")
    if not isinstance(message, str):
        message = str(message)
    if code == "oauth":
        return _build_oauth_text_chunk(message)
    return ExecutorError(
        message=f"Supervisor error ({code}): {message}",
        code=str(code) if code else "supervisor_error",
        retryable=False,
    )


def _build_oauth_text_chunk(message: str) -> TextChunk:
    """
    Build a TextChunk from an OAuth error message.

    Parses the connector name and login URL out of the standard
    Databricks Supervisor OAuth message via :data:`_OAUTH_LOGIN_RE`.
    Falls back to the raw message when the regex does not match
    (some message variant we don't yet know) so the user still
    sees the actionable text.

    The URL is on its own line as plain text. Markdown link syntax
    (``[text](url)`` and ``<url>`` autolinks) both break through
    the TUI streaming pipeline — the raw text is streamed first,
    then the paragraph Markdown re-render fires, producing garbled
    output from the overlap. Plain text avoids the issue entirely.

    :param message: The raw ``message`` field from the error
        event.
    :returns: A :class:`TextChunk` with the login instructions.
    """
    match = _OAUTH_LOGIN_RE.search(message)
    if match is None:
        return TextChunk(text=f"\n\n[auth required] {message}\n\n")
    connector = match.group("connector")
    url = match.group("url")
    return TextChunk(
        text=f"\n\nAuth required - please log in to {connector}:\n{url}\n\n",
    )


def _parse_tool_arguments(item: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a function_call item's ``arguments`` field into a dict.

    The gateway typically emits a JSON-stringified object. Three
    failure modes are handled:

    - Missing field → ``{}`` (the model passed no arguments).
    - Already-decoded dict → returned verbatim.
    - Non-JSON string → wrapped as ``{"raw": <text>}`` so the
      observation carries the bytes the model produced even when
      the gateway's serializer hiccupped.

    :param item: The ``function_call`` item from the SSE event.
    :returns: A plain ``dict`` of argument values.
    """
    raw = item.get("arguments")
    if raw is None or raw == "":
        # Empty / missing arguments are valid — many tools take
        # zero arguments, e.g. ``list_my_drive_files()``.
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Preserve the raw text under a known key so log readers
            # can see what the gateway actually sent. This is a real
            # supervisor-side bug whenever it fires, so log loud — a
            # silent ``{"raw": ...}`` substitution shows up downstream
            # as a phantom dict the model never produced.
            _logger.warning(
                "supervisor function_call arguments were not valid JSON; "
                "wrapping raw text under {'raw': ...}. tool=%r call_id=%r raw=%r",
                item.get("name"),
                item.get("call_id") or item.get("id"),
                raw[:500],
            )
            return {"raw": raw}
        if not isinstance(parsed, dict):
            _logger.warning(
                "supervisor function_call arguments JSON-decoded to %s, "
                "expected dict; wrapping under {'raw': ...}. tool=%r call_id=%r",
                type(parsed).__name__,
                item.get("name"),
                item.get("call_id") or item.get("id"),
            )
            return {"raw": json.dumps(parsed, default=str)}
        return parsed
    # Anything else (number, list) is not a valid arguments shape
    # but we still surface it as a string so the observation isn't
    # silently lossy.
    _logger.warning(
        "supervisor function_call arguments had an unexpected type; "
        "wrapping repr under {'raw': ...}. tool=%r call_id=%r type=%s",
        item.get("name"),
        item.get("call_id") or item.get("id"),
        type(raw).__name__,
    )
    return {"raw": json.dumps(raw, default=str)}


def _extract_tool_result(item: dict[str, Any]) -> str:
    """
    Extract a function_call item's ``result`` field as a string.

    The captured fixture never sets ``result`` on the
    ``function_call`` item itself (the supervisor runs the tool
    internally and inlines the result on a sibling
    ``function_call_output`` item we don't currently consume). When
    a future gateway variant or different tool type DOES populate
    ``result`` here, forward it verbatim.

    :param item: The ``function_call`` item from the SSE event.
    :returns: The result text. Empty string when no result is
        present (the canonical "invocation observed, result not
        yet inlined" state).
    """
    raw = item.get("result")
    if raw is None:
        # Fixture-canonical case: server runs the tool, inlines the
        # output elsewhere, and leaves this field unset.
        return ""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, default=str)


def _map_tool_status(item: dict[str, Any]) -> str:
    """
    Map the gateway's tool ``status`` string to the
    :class:`ToolCallObserved` vocabulary.

    Mapping: ``completed`` → ``success``, ``failed`` → ``error``;
    every other gateway status passes through verbatim so unknown
    states are visible rather than silently mapped.

    :param item: The ``function_call`` item from the SSE event.
    :returns: One of ``{"success", "error", "blocked"}`` (the
        :class:`ToolCallObserved` vocabulary), or the raw upstream
        status when it doesn't match a known gateway value.
    """
    raw = item.get("status")
    if not isinstance(raw, str) or not raw:
        # Missing status is a malformed event; surface as ``error``
        # so the observation isn't reported as a silent success.
        return "error"
    return {"completed": "success", "failed": "error"}.get(raw, raw)


def _resolve_tool_call_id(item: dict[str, Any]) -> str:
    """
    Pull a stable call id out of a function_call item.

    The gateway always emits ``call_id`` in the captured fixture; a
    fallback to ``id`` covers older payload shapes. Both fields
    missing is a malformed event — raise rather than synthesize a
    UUID, because a fabricated id would silently break correlation
    with any later ``function_call_output`` item the gateway emits.

    :param item: The ``function_call`` item from the SSE event.
    :returns: The non-empty call id.
    :raises ValueError: When the item carries neither ``call_id``
        nor ``id``.
    """
    raw = item.get("call_id") or item.get("id")
    if isinstance(raw, str) and raw:
        return raw
    raise ValueError(
        "Supervisor function_call item missing both 'call_id' and 'id'; "
        "cannot correlate observations with downstream tool-result events. "
        f"Item: {item!r}"
    )


def _translate_response_completed(
    payload: dict[str, Any],
    *,
    streamed_text: str | None = None,
) -> TurnComplete:
    """
    Translate a ``response.completed`` event into :class:`TurnComplete`.

    Pulls usage, response id, and finish reasons out of the
    ``payload["response"]`` sub-object. If the response's ``output``
    array contains no text (a real gateway shape — the captured
    fixture exhibits it on healthy turns), fall back to *streamed_text*
    so the workflow's invariant ``TurnComplete.text == join(TextChunks)``
    holds. Without this fallback, persistence would over-write the
    streamed assistant text with ``None``.

    :param payload: The parsed SSE event payload, expected to have
        ``payload["response"]`` with the final response object.
    :param streamed_text: Concatenation of every
        ``response.output_text.delta`` seen on this stream. Used
        only when ``payload["response"]["output"]`` carries no text.
    :returns: A populated :class:`TurnComplete`.
    """
    # ``response`` is sometimes absent or non-dict on malformed
    # payloads from upstream proxies. Default to an empty dict so the
    # downstream extractors uniformly see a dict and we never crash
    # on the terminal event (the workflow needs a TurnComplete to
    # progress).
    response_raw = payload.get("response")
    response: dict[str, Any] = response_raw if isinstance(response_raw, dict) else {}
    text = _extract_completed_text(response)
    if text is None:
        # Empty-output terminator path: the gateway emits
        # response.completed with output=[] even though text deltas
        # streamed. Use the captured deltas so the assistant text is
        # not lost on persist.
        text = streamed_text
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
    response_model = response.get("model") if isinstance(response.get("model"), str) else None
    response_id = response.get("id") if isinstance(response.get("id"), str) else None
    finish_reasons = _extract_finish_reasons(response)
    return TurnComplete(
        text=text,
        usage=usage,
        response_model=response_model,
        response_id=response_id,
        finish_reasons=finish_reasons,
    )


def _extract_completed_text(response: dict[str, Any]) -> str | None:
    """
    Pull the assistant text out of a completed response object.

    Walks ``response["output"]`` for ``message``-typed items and
    concatenates their ``output_text`` content parts. Returns
    ``None`` when no text was emitted (e.g. the turn ended with
    only tool calls).

    :param response: The ``response`` sub-object of a
        ``response.completed`` event.
    :returns: The concatenated assistant text, or ``None`` when
        no text was emitted.
    """
    output = response.get("output")
    if not isinstance(output, list):
        return None
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for piece in content:
            if not isinstance(piece, dict):
                continue
            if piece.get("type") in ("output_text", "text"):
                text = piece.get("text")
                if isinstance(text, str):
                    parts.append(text)
    if not parts:
        return None
    return "".join(parts)


def _extract_finish_reasons(response: dict[str, Any]) -> list[str] | None:
    """
    Pull a finish-reasons list out of a completed response object.

    The supervisor gateway uses ``response["status"]`` (one of
    ``"completed"``, ``"failed"``, etc.) as the closest analogue to
    OpenAI's ``finish_reason``. Returns a single-element list so
    downstream consumers can treat absence (``None``) as "unknown"
    per the :class:`TurnComplete` contract.

    :param response: The ``response`` sub-object of a
        ``response.completed`` event.
    :returns: A single-element list, or ``None`` when no status was
        reported.
    """
    status = response.get("status")
    if isinstance(status, str) and status:
        return [status]
    return None
