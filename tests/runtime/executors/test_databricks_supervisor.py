"""Unit tests for ``omnigent.inner.databricks_supervisor_gateway``.

Each test drives :class:`SupervisorExecutor` against a real
:class:`httpx.AsyncClient` whose transport is a
:class:`httpx.MockTransport`. The transport returns hand-built SSE
byte streams (assembled via :func:`_sse_payload`) so the executor
exercises the same parser path it would in production. No
``MagicMock`` is used for any data object — the SSE payloads are
plain JSON dicts and the resulting events are real
:class:`ExecutorEvent` dataclasses.

Fixture shapes match the real Databricks Supervisor API captured
from staging:

- Tool entries use the NESTED shape (``{type, <type>: {...}}``).
- ``response.output_item.done`` carries an ``item.type ==
  "function_call"`` for tool invocations (not ``<type>_call``).
- OAuth errors arrive as TOP-LEVEL ``{"type": "error", "code":
  "oauth", ...}`` events (NOT nested under ``error.code``).
- The lying terminator: an ``error`` event is followed by
  ``response.completed`` with ``status: "completed"`` and
  ``error: null``. The executor must suppress the success
  TurnComplete in that case.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from omnigent.inner.databricks_supervisor_gateway import (
    GATEWAY_PATH,
    RESPONSES_PATH,
    SupervisorExecutor,
)
from omnigent.runtime.credentials.databricks import WorkspaceCreds
from omnigent.runtime.executors.base import (
    ExecutorContext,
    ExecutorError,
    ExecutorEvent,
    TextChunk,
    ToolCallObserved,
    ToolCallRequested,
    ToolResult,
    TurnComplete,
)
from omnigent.spec.types import (
    AgentSpec,
    ExecutorSpec,
    LLMConfig,
)

# ── Helpers ────────────────────────────────────────────────────────


@dataclass
class _CapturedRequest:
    """
    Snapshot of the request the executor sent to the gateway.

    Used by tests that need to assert on the wire-level shape
    (URL, body, headers) rather than just the response stream.

    :param url: The full request URL.
    :param body: The parsed JSON body.
    :param headers: The request headers as a flat dict.
    """

    url: str
    body: dict[str, Any]
    headers: dict[str, str]


def _sse_payload(events: list[dict[str, Any]]) -> bytes:
    """
    Build an SSE byte stream from a list of JSON events.

    Each event is emitted as ``data: <json>\\n\\n``, matching the
    framing the real Databricks Supervisor gateway emits. No
    ``[DONE]`` terminator is appended — the
    ``response.completed`` event terminates a healthy stream by
    itself.

    :param events: List of JSON-serializable event dicts.
    :returns: The SSE-formatted byte stream ready for the
        :class:`httpx.MockTransport` response body.
    """
    parts: list[str] = []
    for event in events:
        parts.append(f"data: {json.dumps(event)}\n\n")
    return "".join(parts).encode("utf-8")


def _make_supervisor_spec(
    *,
    profile: str | None = None,
    connection: dict[str, str] | None = None,
    supervisor_tools: list[dict[str, Any]] | None = None,
    model: str = "databricks-claude-sonnet-4-6",
) -> AgentSpec:
    """
    Build a minimal :class:`AgentSpec` configured for the supervisor
    executor.

    :param profile: Value for ``executor.profile``.
    :param connection: Value for ``executor.connection``.
    :param supervisor_tools: Value for
        ``executor.supervisor_tools`` (nested shape).
    :param model: Value for ``executor.model``.
    :returns: A populated :class:`AgentSpec` ready for
        :meth:`SupervisorExecutor.from_spec`.
    """
    return AgentSpec(
        spec_version=1,
        name="test-supervisor",
        executor=ExecutorSpec(
            type="omnigent",
            config={"harness": "databricks_supervisor"},
            model=model,
            connection=connection,
            profile=profile,
            supervisor_tools=supervisor_tools,
        ),
        llm=LLMConfig(model=model, connection=connection),
    )


def _build_executor_with_capture(
    captured: list[_CapturedRequest],
    response_events: list[dict[str, Any]] | None = None,
    *,
    status_code: int = 200,
    raw_body: bytes | None = None,
    spec: AgentSpec | None = None,
) -> SupervisorExecutor:
    """
    Build a :class:`SupervisorExecutor` whose http client returns a
    pre-built SSE response and records each outbound request into
    *captured*.

    :param captured: List that the mock transport appends each
        observed request to.
    :param response_events: If given, the events to encode into the
        response body via :func:`_sse_payload`.
    :param status_code: HTTP status code to return.
    :param raw_body: If provided, used as the response body
        verbatim (overrides *response_events*).
    :param spec: Optional pre-built spec; default builds a minimal
        spec with explicit connection so no network resolution is
        attempted.
    :returns: A configured executor whose ``http_client`` is the
        mock client.
    """
    if raw_body is None:
        raw_body = _sse_payload(response_events or [])

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured.append(
            _CapturedRequest(
                url=str(request.url),
                body=body,
                headers=dict(request.headers.items()),
            )
        )
        return httpx.Response(
            status_code=status_code,
            content=raw_body,
            headers={"Content-Type": "text/event-stream"},
        )

    transport = httpx.MockTransport(_handler)
    client = httpx.AsyncClient(transport=transport)
    if spec is None:
        spec = _make_supervisor_spec(
            connection={
                "base_url": "https://example.test/ai-gateway/mlflow/v1",
                "api_key": "test-token",
            },
        )
    assert spec.executor.model is not None  # _make_supervisor_spec always populates it
    connection = spec.executor.connection or {}
    base = connection.get("base_url") or "https://example.test/ai-gateway/mlflow/v1"
    return SupervisorExecutor(
        model=spec.executor.model,
        supervisor_tools=spec.executor.supervisor_tools,
        base_url=base.rstrip("/"),
        api_key=connection.get("api_key", "test-token"),
        http_client=client,
    )


async def _collect(stream: AsyncIterator[ExecutorEvent]) -> list[ExecutorEvent]:
    """
    Drain an async iterator of executor events into a list.

    :param stream: The async iterator to consume.
    :returns: All yielded items in order.
    """
    out: list[ExecutorEvent] = []
    async for item in stream:
        out.append(item)
    return out


def _terminator(
    *,
    response_id: str = "resp_test",
    sequence_number: int | None = None,
    output: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a standard gateway ``response.completed`` SSE event.

    Almost every test fixture ends with this event; centralising it
    keeps the per-test event lists focused on the events the test
    actually asserts on.

    :param response_id: The ``response.id`` field, e.g. ``"resp_abc"``.
    :param sequence_number: Optional sequence number to set on the
        event; omitted from the dict when ``None``.
    :param output: The ``response.output`` array. Defaults to ``[]``.
    :returns: The event dict ready to drop into a ``response_events``
        list.
    """
    event: dict[str, Any] = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "status": "completed",
            "output": output if output is not None else [],
            "error": None,
        },
    }
    if sequence_number is not None:
        event["sequence_number"] = sequence_number
    return event


async def _run_and_collect(
    executor: SupervisorExecutor,
    *,
    llm_config: LLMConfig,
    context: ExecutorContext,
    prompt: str = "hi",
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
) -> list[ExecutorEvent]:
    """
    Run a single turn against *executor* and collect every event.

    Wraps the run_turn(messages=[{...}], tools=..., ...) + _collect
    boilerplate that nearly every streaming test repeats.

    :param executor: A :class:`SupervisorExecutor` (typically built
        via :func:`_build_executor_with_capture`).
    :param llm_config: The standard ``llm_config`` fixture.
    :param context: The standard ``exec_context`` fixture.
    :param prompt: The single user message to send,
        e.g. ``"search drive"``.
    :param tools: Optional workflow-supplied tools list. Defaults to
        ``[]`` (the supervisor ignores it anyway).
    :param system_prompt: System prompt; defaults to ``""``.
    :returns: All :class:`ExecutorEvent` instances yielded by the turn,
        in order.
    """
    return await _collect(
        executor.run_turn(
            messages=[{"role": "user", "content": prompt}],
            tools=tools if tools is not None else [],
            system_prompt=system_prompt,
            llm_config=llm_config,
            context=context,
        )
    )


@pytest.fixture()
def llm_config() -> LLMConfig:
    """
    Minimal :class:`LLMConfig` for ``run_turn`` calls.

    The supervisor executor reads ``model`` from the spec at
    construction time, not from this :class:`LLMConfig`, so this
    fixture only needs to satisfy the ABC.
    """
    return LLMConfig(model="databricks-claude-sonnet-4-6")


@pytest.fixture()
def exec_context(tmp_path: Path) -> ExecutorContext:
    """
    :class:`ExecutorContext` whose ``call_tool`` raises if invoked.

    The supervisor executor never dispatches tools through the
    workflow (server-runs-tools idiom), so any invocation is a bug.
    """

    async def _unused_call_tool(req: ToolCallRequested) -> ToolResult:
        raise AssertionError(
            "SupervisorExecutor must not dispatch tools via "
            "context.call_tool — server-runs-tools idiom."
        )

    async def _unused_enforce(_name: str, _args: dict[str, Any]) -> str | None:
        raise AssertionError(
            "SupervisorExecutor must not invoke "
            "enforce_tool_call_policy — gateway runs tools "
            "internally, no AP-side dispatch."
        )

    return ExecutorContext(
        task_id="task_test_123",
        conversation_id="conv_test_456",
        storage_dir=tmp_path,
        call_tool=_unused_call_tool,
        enforce_tool_call_policy=_unused_enforce,
    )


# ── from_spec / construction ──────────────────────────────────────


def _build_capture_client(
    captured: list[_CapturedRequest],
) -> httpx.AsyncClient:
    """
    Build a :class:`httpx.AsyncClient` whose mock transport records
    each outbound request to *captured* and returns a no-op success
    SSE stream.

    Tests pass the result to :meth:`SupervisorExecutor.from_spec`'s
    ``http_client`` kwarg so they can assert on the eventual request
    URL through the public entry point (no poking of private state).

    :param captured: List that receives one
        :class:`_CapturedRequest` when the executor runs a turn.
    :returns: An async client wrapping a :class:`httpx.MockTransport`.
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        captured.append(
            _CapturedRequest(
                url=str(request.url),
                body=body,
                headers=dict(request.headers.items()),
            )
        )
        return httpx.Response(
            status_code=200,
            content=_sse_payload(
                [
                    {
                        "type": "response.completed",
                        "sequence_number": 0,
                        "response": {
                            "id": "resp_x",
                            "status": "completed",
                            "output": [],
                            "error": None,
                        },
                    }
                ]
            ),
            headers={"Content-Type": "text/event-stream"},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(_handler))


@pytest.mark.asyncio
async def test_from_spec_with_profile_resolves_gateway_url(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: LLMConfig,
    exec_context: ExecutorContext,
) -> None:
    """
    Spec without explicit base_url + api_key falls back to the
    Databricks credentials resolver, and the gateway URL is composed
    from the resolved host plus :data:`GATEWAY_PATH`. Asserted via
    the eventual request URL (a public observable) so the test is
    not coupled to internal attribute names.
    """
    captured_profiles: list[str | None] = []

    def _fake_resolve(profile: str | None) -> WorkspaceCreds:
        captured_profiles.append(profile)
        return WorkspaceCreds(
            host="https://example.databricks.com",
            token="resolved-token",
        )

    monkeypatch.setattr(
        "omnigent.inner.databricks_supervisor_gateway.resolve_databricks_workspace",
        _fake_resolve,
    )
    spec = _make_supervisor_spec(profile="test-profile")

    captured: list[_CapturedRequest] = []
    executor = SupervisorExecutor.from_spec(spec, http_client=_build_capture_client(captured))
    await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
    )

    # Resolver was called with the profile from the spec — proves
    # the executor is reading from the concrete profile field, not
    # silently substituting a default.
    assert captured_profiles == ["test-profile"]
    # URL == resolved host + GATEWAY_PATH + RESPONSES_PATH —
    # verifies the gateway path is appended literally and the
    # resolver's host (which strips trailing slashes) was used
    # verbatim.
    assert len(captured) == 1
    expected_url = f"https://example.databricks.com{GATEWAY_PATH}{RESPONSES_PATH}"
    assert captured[0].url == expected_url


@pytest.mark.asyncio
async def test_from_spec_with_explicit_connection_uses_it(
    monkeypatch: pytest.MonkeyPatch,
    llm_config: LLMConfig,
    exec_context: ExecutorContext,
) -> None:
    """
    Explicit ``llm.connection`` (both base_url and api_key) wins
    over profile resolution. Asserted via the request URL (public
    observable) so the test is not coupled to private attribute
    names.
    """

    def _fake_resolve(profile: str | None) -> WorkspaceCreds:
        raise AssertionError(
            "resolve_databricks_workspace must NOT be called when "
            "spec.executor.connection has both base_url and api_key — "
            "explicit connection should short-circuit resolution."
        )

    monkeypatch.setattr(
        "omnigent.inner.databricks_supervisor_gateway.resolve_databricks_workspace",
        _fake_resolve,
    )
    spec = _make_supervisor_spec(
        profile="test-profile",
        connection={
            "base_url": "https://override.test/ai-gateway/mlflow/v1",
            "api_key": "explicit-token",
        },
    )

    captured: list[_CapturedRequest] = []
    executor = SupervisorExecutor.from_spec(spec, http_client=_build_capture_client(captured))
    await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
    )

    # URL uses the explicit base_url verbatim — proves the
    # connection-block branch fired and resolver was bypassed
    # (the resolver above also asserts if reached).
    assert len(captured) == 1
    assert captured[0].url == (f"https://override.test/ai-gateway/mlflow/v1{RESPONSES_PATH}")


@pytest.mark.parametrize(
    ("connection", "missing"),
    [
        ({"base_url": "https://x.example.com/ai-gateway/mlflow/v1"}, "api_key"),
        ({"api_key": "tok-only"}, "base_url"),
    ],
    ids=["only_base_url", "only_api_key"],
)
def test_from_spec_partial_connection_raises(connection: dict[str, str], missing: str) -> None:
    """
    Partially populated ``llm.connection`` (one of base_url/api_key
    set, the other missing) MUST raise rather than silently fall
    back to the resolver. Without this check, a spec that intends
    to override the workspace would send the supervisor to ambient
    creds, causing very confusing cross-workspace bugs.
    """
    spec = _make_supervisor_spec(connection=connection)

    with pytest.raises(ValueError) as excinfo:
        SupervisorExecutor.from_spec(spec)

    msg = str(excinfo.value)
    # The error should name BOTH the populated key AND the missing
    # one so the user knows exactly what to fix.
    assert missing in msg
    other = "api_key" if missing == "base_url" else "base_url"
    assert other in msg


async def test_run_turn_honors_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
    exec_context: ExecutorContext,
) -> None:
    """
    ``llm_config.request_timeout`` must flow into the ``read``
    timeout of the httpx client. Without this, a spec lowering the
    timeout to fail fast would silently wait the 600s default
    instead.
    """
    captured_timeouts: list[httpx.Timeout] = []

    class _RecordingClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            timeout = kwargs.get("timeout")
            if isinstance(timeout, httpx.Timeout):
                captured_timeouts.append(timeout)
            # Force a quick failure so the test doesn't actually try
            # to talk to a network — we only care about the timeout
            # configuration being passed through.
            kwargs["transport"] = httpx.MockTransport(
                lambda _req: httpx.Response(503, content=b"")
            )
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        "omnigent.inner.databricks_supervisor_gateway.httpx.AsyncClient",
        _RecordingClient,
    )

    spec = _make_supervisor_spec(
        connection={
            "base_url": "https://x.example.com/ai-gateway/mlflow/v1",
            "api_key": "tok",
        },
    )
    # NOTE: this path constructs its own AsyncClient (no
    # http_client kwarg) so we exercise the production timeout
    # construction code.
    executor = SupervisorExecutor.from_spec(spec)
    cfg = LLMConfig(model="databricks-claude-sonnet-4-6", request_timeout=42)

    await _run_and_collect(
        executor,
        llm_config=cfg,
        context=exec_context,
    )

    # One client construction → one Timeout object captured. The
    # read timeout should match the spec's request_timeout exactly,
    # proving the field flowed through. Connect/pool stay at the
    # conservative module default.
    assert len(captured_timeouts) == 1
    assert captured_timeouts[0].read == pytest.approx(42.0)
    assert captured_timeouts[0].write == pytest.approx(42.0)


def test_max_context_tokens_returns_none() -> None:
    """
    The supervisor's server-side loop is opaque, so the workflow
    must skip compaction. ``max_context_tokens()`` returns
    ``None`` to signal that.
    """
    spec = _make_supervisor_spec(
        connection={
            "base_url": "https://example.test/ai-gateway/mlflow/v1",
            "api_key": "tok",
        },
    )
    executor = SupervisorExecutor.from_spec(spec)

    assert executor.max_context_tokens() is None


# ── run_turn streaming ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_streams_text_chunks(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    Three ``response.output_text.delta`` events produce three
    ``TextChunk`` events with the matching deltas.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            {"type": "response.created", "sequence_number": 0, "response": {}},
            {"type": "response.in_progress", "sequence_number": 1, "response": {}},
            {
                "type": "response.output_text.delta",
                "delta": "Hello ",
                "sequence_number": 2,
            },
            {
                "type": "response.output_text.delta",
                "delta": "world",
                "sequence_number": 3,
            },
            {
                "type": "response.output_text.delta",
                "delta": "!",
                "sequence_number": 4,
            },
            {
                "type": "response.completed",
                "sequence_number": 5,
                "response": {
                    "id": "resp_abc",
                    "model": "databricks-claude-sonnet-4-6",
                    "status": "completed",
                    "output": [],
                    "error": None,
                },
            },
        ],
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
        system_prompt="be helpful",
    )

    text_chunks = [e for e in events if isinstance(e, TextChunk)]
    # Exactly 3 TextChunks — proves each delta event mapped 1:1 and
    # the response.created / response.in_progress events were
    # silently consumed (no extra TextChunks synthesized).
    assert len(text_chunks) == 3, (
        f"Expected 3 TextChunks (one per delta event), got "
        f"{len(text_chunks)}. If 4+, response.created or "
        f"response.in_progress incorrectly produced a chunk."
    )
    assert [c.text for c in text_chunks] == ["Hello ", "world", "!"]


@pytest.mark.asyncio
async def test_run_turn_emits_tool_call_observed_without_request(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    Server-executed tool result event yields a ``ToolCallObserved``
    and NO ``ToolCallRequested`` precedes it (server-runs-tools
    idiom).

    Real fixture shape from staging: ``response.output_item.done``
    with ``item.type == "function_call"`` carries the tool name +
    arguments + call_id.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            {
                "type": "response.output_item.done",
                "sequence_number": 7,
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "msg_bdrk_01",
                    "call_id": "toolu_bdrk_01DbjFubYEDiva1b1vaDxkMi",
                    "name": "google_drive_search",
                    "arguments": (
                        '{"query": "modifiedTime > \'2020-01-01T00:00:00\'", "max_results": 3}'
                    ),
                    "status": "completed",
                },
            },
            _terminator(response_id="resp_def", sequence_number=8),
        ],
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
        prompt="search drive",
    )

    # No ToolCallRequested emitted — server runs the tool, Omnigent just
    # observes the result.
    assert not any(isinstance(e, ToolCallRequested) for e in events), (
        "SupervisorExecutor must NOT emit ToolCallRequested — the "
        "supervisor gateway runs every tool itself; Omnigent only "
        "observes the result."
    )
    observed = [e for e in events if isinstance(e, ToolCallObserved)]
    # Exactly one ToolCallObserved — proves the function_call item
    # mapped 1:1 to a single observation.
    assert len(observed) == 1, (
        f"Expected exactly 1 ToolCallObserved (one function_call "
        f"in the stream), got {len(observed)}."
    )
    obs = observed[0]
    # ``name`` is the function name straight from the item, NOT a
    # transformed item.type — proves we read the right field.
    assert obs.name == "google_drive_search"
    assert obs.call_id == "toolu_bdrk_01DbjFubYEDiva1b1vaDxkMi"
    # arguments JSON-decoded into a dict.
    assert obs.arguments == {
        "query": "modifiedTime > '2020-01-01T00:00:00'",
        "max_results": 3,
    }
    # gateway "completed" → Omnigent "success" status mapping.
    assert obs.status == "success"


# ── _map_tool_status: full table coverage ─────────────────────────


@pytest.mark.parametrize(
    "gateway_status,expected_observed",
    [
        # Happy path — completed must collapse to "success".
        ("completed", "success"),
        # Explicit failed must surface as "error" so policy /
        # retry layers can distinguish from success.
        ("failed", "error"),
        # Missing status string (None) is malformed — must map to
        # "error" so a silent success-coded surprise can't slip
        # through. Regression for a swap like
        # ``{"failed": "success"}`` that the happy-path test alone
        # would not catch.
        (None, "error"),
        # Empty string is also malformed → "error".
        ("", "error"),
        # Unknown gateway statuses pass through verbatim — the
        # workflow consumer will treat them as a non-success
        # observation, but we don't fabricate a mapping.
        ("running", "running"),
        ("blocked", "blocked"),
    ],
)
@pytest.mark.asyncio
async def test_function_call_status_mapping_full_table(
    gateway_status: str | None,
    expected_observed: str,
    llm_config: LLMConfig,
    exec_context: ExecutorContext,
) -> None:
    """
    Exercise the full mapping table the gateway can produce on a
    ``function_call`` item. The happy-path test only covered
    ``completed → success``; this guards against regressions in
    the unknown-passthrough and missing/empty-status branches that
    would otherwise be free to drift.
    """
    item: dict[str, Any] = {
        "type": "function_call",
        "id": "msg_status_x",
        "call_id": "toolu_status_x",
        "name": "noop",
        "arguments": "{}",
    }
    if gateway_status is not None:
        item["status"] = gateway_status
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            {
                "type": "response.output_item.done",
                "sequence_number": 1,
                "item": item,
            },
            _terminator(response_id="resp_status_x", sequence_number=2),
        ],
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
        prompt="noop",
    )

    observed = [e for e in events if isinstance(e, ToolCallObserved)]
    assert len(observed) == 1
    assert observed[0].status == expected_observed


@pytest.mark.asyncio
async def test_function_call_output_populates_tool_result(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    The Supervisor gateway emits the tool invocation and its result
    on SEPARATE SSE events: a ``response.output_item.done`` carrying
    a ``function_call`` item, then a sibling
    ``response.output_item.done`` carrying a ``function_call_output``
    item. The executor MUST pair the two by ``call_id`` and emit a
    SINGLE :class:`ToolCallObserved` whose ``result`` field carries
    the output text. Without pairing, every tool call would persist
    with ``result == ""`` and follow-up turns / ``/history`` /
    reconnect would all see empty results.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            # Tool invocation
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "msg_a",
                    "call_id": "toolu_pairing_test",
                    "name": "list_files",
                    "arguments": '{"q": "*"}',
                    "status": "completed",
                },
            },
            # Sibling output event with the result text
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call_output",
                    "call_id": "toolu_pairing_test",
                    "output": "found 3 files: a, b, c",
                    "status": "completed",
                },
            },
            _terminator(response_id="resp_pair"),
        ],
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
        prompt="list",
    )

    observed = [e for e in events if isinstance(e, ToolCallObserved)]
    # EXACTLY one ToolCallObserved — the function_call invocation
    # must be buffered, not emitted twice. If this is 2, we are
    # double-emitting (once on function_call.done, once on
    # function_call_output.done) and downstream consumers will see
    # the same call twice.
    assert len(observed) == 1, (
        f"Expected exactly 1 ToolCallObserved (paired); got {len(observed)}."
    )
    obs = observed[0]
    assert obs.call_id == "toolu_pairing_test"
    assert obs.name == "list_files"
    # The result field MUST come from the function_call_output's
    # ``output`` field. If it's empty, the pairing logic regressed
    # and persistence will lose tool outputs.
    assert obs.result == "found 3 files: a, b, c"


@pytest.mark.asyncio
async def test_function_call_without_output_event_flushes_at_stream_end(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    When a turn ends WITHOUT the function_call_output event firing
    (OAuth-required, gateway hiccup, mid-stream cancellation), the
    buffered invocation must still surface as a
    :class:`ToolCallObserved` — with ``result == ""`` to mark
    "invocation observed but result not received." Otherwise the
    tool call disappears entirely from history.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "msg_b",
                    "call_id": "toolu_unpaired",
                    "name": "missing_output",
                    "arguments": "{}",
                    "status": "completed",
                },
            },
            # NOTE: no function_call_output event. Gateway just
            # sends the terminator next.
            _terminator(response_id="resp_x"),
        ],
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
        prompt="x",
    )

    observed = [e for e in events if isinstance(e, ToolCallObserved)]
    assert len(observed) == 1, (
        "function_call without matching output must still emit "
        "exactly one ToolCallObserved at stream end so the "
        "invocation isn't silently dropped."
    )
    obs = observed[0]
    assert obs.call_id == "toolu_unpaired"
    assert obs.name == "missing_output"
    # Empty result marks "invocation observed, output not received."
    assert obs.result == ""


@pytest.mark.asyncio
async def test_function_call_missing_call_id_raises(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    A ``function_call`` item missing BOTH ``call_id`` and ``id``
    must surface as a ValueError, not be silently swallowed with a
    synthesized UUID. A fabricated id would break correlation with
    any later ``function_call_output`` item the gateway may emit
    on this stream — the observation would land in omnigent
    history but the result would never link back to it.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "name": "google_drive_search",
                    "arguments": "{}",
                    "status": "completed",
                    # NOTE: no ``call_id`` and no ``id``
                },
            },
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        await _run_and_collect(
            executor,
            llm_config=llm_config,
            context=exec_context,
            prompt="x",
        )

    msg = str(excinfo.value)
    # The error names BOTH missing keys so the user knows exactly
    # which fields the gateway omitted.
    assert "call_id" in msg
    assert "id" in msg


@pytest.mark.asyncio
async def test_run_turn_oauth_required_emits_text_chunk_with_login_url(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    OAuth-required event (top-level ``error`` with ``code: oauth``)
    produces a TextChunk containing the connector name and login
    URL parsed from the message.

    Uses the EXACT message format captured from staging so the
    regex faces real data, not a synthetic format.
    """
    captured: list[_CapturedRequest] = []
    real_oauth_message = (
        "Error searching Google Drive: Failed request to "
        "https://www.googleapis.com:443/drive/v3/files. Error: "
        "Credential for user identity('1234567890') is not "
        "found for the connection 'system_ai_agent_google_drive'. "
        "Please login first to the connection by visiting "
        "https://example.databricks.com/"
        "explore/connections/system_ai_agent_google_drive?o=12345"
    )
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            {
                "type": "error",
                "code": "oauth",
                "message": real_oauth_message,
                "sequence_number": 9,
            },
            # The lying terminator — must be suppressed because the
            # error fired first.
            _terminator(response_id="resp_ghi", sequence_number=10),
        ],
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
        prompt="search drive",
    )

    text_chunks = [e for e in events if isinstance(e, TextChunk)]
    # Exactly one TextChunk — proves the OAuth event mapped 1:1.
    # (The lying success terminator that follows the error is
    # swallowed; we don't emit a TextChunk for it.)
    assert len(text_chunks) == 1
    text = text_chunks[0].text
    # Both the connector name and URL are present — proves the
    # regex parsed the real-staging message and both groups landed
    # in the output.
    assert "system_ai_agent_google_drive" in text
    assert (
        "https://example.databricks.com/explore/connections/system_ai_agent_google_drive?o=12345"
    ) in text
    assert "Auth required" in text
    assert text.startswith("\n\n")
    assert text.endswith("\n\n")
    # A SYNTHESIZED TurnComplete must follow the OAuth TextChunk,
    # carrying the auth message as its ``text`` field. Without this,
    # the workflow's reconnect/history paths would derive the final
    # response text from a missing TurnComplete and persist a blank
    # assistant message after the auth prompt. The lying success
    # terminator from the gateway is still suppressed (no usage /
    # response_id surfaces from it).
    turn_completes = [e for e in events if isinstance(e, TurnComplete)]
    assert len(turn_completes) == 1, (
        "OAuth-required path must emit exactly one synthesized "
        "TurnComplete carrying the auth message; got "
        f"{len(turn_completes)}"
    )
    assert turn_completes[0].text == text, (
        "Synthesized TurnComplete.text must match the auth-required "
        "TextChunk so persistence stores the actionable login URL."
    )
    assert turn_completes[0].usage is None
    assert turn_completes[0].response_id is None


@pytest.mark.asyncio
async def test_run_turn_empty_output_completion_uses_streamed_text(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    A healthy turn whose terminator carries an empty
    ``response.output`` array MUST still produce a
    :class:`TurnComplete` whose ``text`` matches the concatenated
    streamed text deltas.

    This shape (text deltas streamed earlier, terminator's output
    empty) was captured from the real gateway. Without the
    streamed-text fallback in
    :func:`_translate_response_completed`, the workflow's
    persistence layer would over-write the streamed assistant text
    with ``None`` because :class:`TurnComplete.text` was the source
    of truth.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            {
                "type": "response.output_text.delta",
                "delta": "Hello, ",
            },
            {
                "type": "response.output_text.delta",
                "delta": "world!",
            },
            # Terminator with EMPTY output array — exactly the
            # shape the captured staging fixture exhibits on
            # otherwise healthy turns.
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_empty",
                    "status": "completed",
                    "output": [],
                    "model": "databricks-claude-sonnet-4-6",
                },
            },
        ],
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
        prompt="say hi",
    )

    # The TurnComplete's text is the concatenation of the two
    # delta values, NOT None. If this returned None, the empty-
    # output fallback regressed and persistence would lose the
    # streamed assistant text.
    final = next(e for e in events if isinstance(e, TurnComplete))
    assert final.text == "Hello, world!"
    # The metadata still flows from the terminator, just not the
    # text.
    assert final.response_id == "resp_empty"
    assert final.response_model == "databricks-claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_run_turn_terminates_with_turn_complete_on_success(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    Happy-path stream's final event is :class:`TurnComplete`.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[
            {
                "type": "response.output_text.delta",
                "delta": "Done.",
                "sequence_number": 0,
            },
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {
                    "id": "resp_jkl",
                    "model": "databricks-claude-sonnet-4-6",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Done."},
                            ],
                        },
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 3},
                    "error": None,
                },
            },
        ],
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
    )

    # Final event MUST be TurnComplete — the workflow's terminal
    # contract; without this the workflow would treat the turn as
    # incomplete and re-issue.
    assert isinstance(events[-1], TurnComplete)
    final = events[-1]
    assert final.text == "Done."
    assert final.response_id == "resp_jkl"
    assert final.response_model == "databricks-claude-sonnet-4-6"
    assert final.usage == {"input_tokens": 12, "output_tokens": 3}
    assert final.finish_reasons == ["completed"]


@pytest.mark.asyncio
async def test_run_turn_emits_executor_error_on_4xx(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    HTTP 400 from the gateway becomes an :class:`ExecutorError`,
    not an exception, so the workflow can persist a clean failure.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        raw_body=b'{"error": "bad request"}',
        status_code=400,
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
    )

    errors = [e for e in events if isinstance(e, ExecutorError)]
    assert len(errors) == 1
    err = errors[0]
    # Status code surfaced in the error code so policy/log code
    # can branch on it.
    assert err.code == "http_400"
    # 4xx is a client/spec error — not retryable.
    assert err.retryable is False
    assert "400" in err.message


@pytest.mark.asyncio
async def test_run_turn_emits_executor_error_on_5xx(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    HTTP 500 becomes a retryable :class:`ExecutorError`.
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        raw_body=b'{"error": "internal"}',
        status_code=500,
    )

    events = await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
    )

    errors = [e for e in events if isinstance(e, ExecutorError)]
    assert len(errors) == 1
    err = errors[0]
    assert err.code == "http_500"
    # 5xx is a server transient — retryable so the surrounding
    # retry policy can reissue.
    assert err.retryable is True


# ── Request body shape ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_body_shape(llm_config: LLMConfig, exec_context: ExecutorContext) -> None:
    """
    Request body includes: model, input (messages verbatim), tools
    (verbatim NESTED form from spec.executor.supervisor_tools),
    instructions (system_prompt), stream=true.
    """
    captured: list[_CapturedRequest] = []
    nested_tools: list[dict[str, Any]] = [
        {
            "type": "genie_space",
            "genie_space": {"id": "abc", "description": "demo space"},
        },
        {
            "type": "uc_function",
            "uc_function": {"name": "main.fn", "description": "demo fn"},
        },
    ]
    spec = _make_supervisor_spec(
        connection={
            "base_url": "https://example.test/ai-gateway/mlflow/v1",
            "api_key": "tok",
        },
        supervisor_tools=nested_tools,
    )
    executor = _build_executor_with_capture(
        captured,
        response_events=[_terminator(response_id="resp_x", sequence_number=0)],
        spec=spec,
    )

    messages = [
        {"role": "user", "content": "what?"},
        {"role": "assistant", "content": "huh?"},
    ]
    await _collect(
        executor.run_turn(
            messages=messages,
            tools=[],
            system_prompt="be terse",
            llm_config=llm_config,
            context=exec_context,
        )
    )

    assert len(captured) == 1
    req = captured[0]
    # URL has the responses path appended to the base_url — proves
    # the executor uses both pieces and didn't accidentally strip
    # the path off.
    assert req.url == f"https://example.test/ai-gateway/mlflow/v1{RESPONSES_PATH}"
    # Body: every required field present and verbatim.
    assert req.body["model"] == "databricks-claude-sonnet-4-6"
    assert req.body["input"] == messages
    # Tools round-trip in the NESTED form — the executor MUST NOT
    # reshape; the gateway rejects flat shapes.
    assert req.body["tools"] == nested_tools
    assert req.body["instructions"] == "be terse"
    assert req.body["stream"] is True


@pytest.mark.asyncio
async def test_request_body_omits_instructions_when_empty(
    llm_config: LLMConfig, exec_context: ExecutorContext
) -> None:
    """
    Empty system prompt → request body has NO ``instructions`` key.

    Empty and absent are semantically identical to the gateway,
    but omitting is cleaner on the wire and matches the spec
    self-containment principle (don't send a field unless it
    means something).
    """
    captured: list[_CapturedRequest] = []
    executor = _build_executor_with_capture(
        captured,
        response_events=[_terminator(response_id="resp_y", sequence_number=0)],
    )

    await _run_and_collect(
        executor,
        llm_config=llm_config,
        context=exec_context,
        prompt="x",
    )

    assert len(captured) == 1
    # The key is absent (not "" or None) so the gateway sees a
    # clean request without a meaningless field.
    assert "instructions" not in captured[0].body
