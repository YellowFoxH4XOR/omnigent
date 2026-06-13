"""
Tests for the ``executor.type: supervisor`` harness wrap.

Mirror of ``tests/inner/test_codex_harness.py`` — verifies the wrap
module has the same shape (registry entry, FastAPI app routes, env-var-
driven lazy executor construction). Does NOT exercise the real
Databricks gateway; the runtime executor's class is replaced with a
stub so the test passes without real workspace credentials.

End-to-end verification (real gateway, real connectors) lives in
``tests/e2e/omnigent/test_run_omnigent_supervisor.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from omnigent.inner import databricks_supervisor_executor, databricks_supervisor_harness
from omnigent.inner.executor import (
    ExecutorError as InnerExecutorError,
)
from omnigent.inner.executor import (
    TextChunk as InnerTextChunk,
)
from omnigent.inner.executor import (
    ToolCallComplete,
    ToolCallRequest,
    ToolCallStatus,
)
from omnigent.inner.executor import (
    TurnComplete as InnerTurnComplete,
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
    TurnComplete as RuntimeTurnComplete,
)
from omnigent.runtime.harnesses import _HARNESS_MODULES


@dataclass
class _CapturedConstruction:
    """
    Recorded constructor kwargs from a stubbed ``SupervisorExecutor``.

    Tests assert against this dataclass instead of an opaque dict
    so a regression that drops a field shows up as a clear
    AttributeError instead of a missing key in some captured
    mapping.

    :param model: The ``model`` kwarg passed to the runtime
        executor.
    :param supervisor_tools: The ``supervisor_tools`` kwarg
        (list of nested-shape tool entries, or ``None``).
    :param base_url: The ``base_url`` kwarg.
    :param api_key: The ``api_key`` kwarg.
    :param http_client: The ``http_client`` kwarg (always
        ``None`` from the wrap; tests verify the wrap doesn't
        accidentally inject one).
    """

    model: str | None = None
    supervisor_tools: list[dict[str, Any]] | None = None
    base_url: str | None = None
    api_key: str | None = None
    http_client: Any = None
    profile: str | None = None  # populated by the resolver stub
    construction_count: int = 0


class _StubRuntimeSupervisorExecutor:
    """
    Drop-in replacement for ``omnigent.inner.databricks_supervisor_gateway.SupervisorExecutor``.

    Mirrors the real constructor's kw-only signature precisely — a
    regression that adds a required kwarg upstream surfaces as a
    TypeError on stub instantiation, not as a silent test pass with
    a captured-dict missing the new key.

    :param model: Forwarded into the captured record.
    :param supervisor_tools: Forwarded.
    :param base_url: Forwarded.
    :param api_key: Forwarded.
    :param http_client: Forwarded; defaults to ``None`` (tests
        verify the wrap doesn't inject one).
    :param captured: The shared :class:`_CapturedConstruction`
        instance the stub writes into. Bound at class-creation
        time (see :meth:`_install_stub`).
    """

    captured: _CapturedConstruction | None = None

    def __init__(
        self,
        *,
        model: str,
        supervisor_tools: list[dict[str, Any]] | None,
        base_url: str,
        api_key: str,
        http_client: Any = None,
    ) -> None:
        # Class-bound captured record. Tests reset this per-test
        # via :func:`_install_stub`. We bump construction_count so
        # tests can spot accidental double construction (the wrap
        # builds lazily; calling it twice in one test is a bug).
        captured = type(self).captured
        if captured is None:  # pragma: no cover — paranoia
            raise RuntimeError("stub captured record not installed; call _install_stub first")
        captured.model = model
        captured.supervisor_tools = supervisor_tools
        captured.base_url = base_url
        captured.api_key = api_key
        captured.http_client = http_client
        captured.construction_count += 1


def _install_stub(monkeypatch: pytest.MonkeyPatch) -> _CapturedConstruction:
    """
    Replace the real runtime ``SupervisorExecutor`` class with the
    stub for the duration of one test.

    Patches at the module that imports it (the inner wrap module's
    name binding), per the standard "patch where it's used, not
    where it's defined" rule. The stub records kwargs into a fresh
    :class:`_CapturedConstruction` per call so tests are isolated.

    :param monkeypatch: pytest's monkeypatch fixture.
    :returns: The captured-record instance the stub will write
        into. Tests assert against this directly.
    """
    captured = _CapturedConstruction()
    _StubRuntimeSupervisorExecutor.captured = captured
    monkeypatch.setattr(
        "omnigent.inner.databricks_supervisor_executor._RuntimeSupervisorExecutor",
        _StubRuntimeSupervisorExecutor,
    )
    return captured


def test_harness_module_registered_in_module_registry() -> None:
    """``"databricks_supervisor"`` resolves to the harness module path.

    Without this entry, the runner subprocess cannot find the wrap
    when AP-side tries to spawn it for a
    ``config.harness: databricks_supervisor`` spec.
    """
    # Locks the registry pointer — a refactor that renamed the wrap
    # module would silently break the dispatch path; this assert
    # surfaces the rename immediately.
    assert (
        _HARNESS_MODULES.get("databricks_supervisor")
        == "omnigent.inner.databricks_supervisor_harness"
    )


def test_create_app_returns_fastapi_with_required_routes() -> None:
    """``create_app()`` returns a FastAPI app exposing the harness API.

    Verifies the wrap successfully:
    - Imports the executor adapter + inner SupervisorExecutor.
    - Builds the FastAPI app via ExecutorAdapter.build().
    - Mounts the standard harness routes.

    The runtime SupervisorExecutor is constructed lazily on the first
    turn (not at app build time), so this test passes without real
    Databricks credentials.
    """
    app = databricks_supervisor_harness.create_app()
    paths = {route.path for route in app.routes}  # type: ignore[attr-defined]
    # Session-keyed harness API: liveness probe + single
    # discriminated-event endpoint per §The Harness API Subset.
    assert "/health" in paths
    assert "/v1/sessions/{conversation_id}/events" in paths


def test_executor_factory_threads_model_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``HARNESS_SUPERVISOR_MODEL`` flows into the runtime executor.

    Locks in the v1 config-flow contract: env vars set in AP's
    process before spawning the subprocess are how the wrap learns
    its config. The model arrives via ``HARNESS_SUPERVISOR_MODEL``
    and must reach the runtime executor's ``model`` kwarg.
    """
    monkeypatch.setenv("HARNESS_SUPERVISOR_MODEL", "databricks-claude-sonnet-4-6")
    # The wrap requires SOME credential source. Wire an explicit
    # connection so the test doesn't try to read ~/.databrickscfg.
    monkeypatch.setenv(
        "HARNESS_SUPERVISOR_CONNECTION_JSON",
        json.dumps(
            {
                "base_url": "https://example.databricks.com/ai-gateway/mlflow/v1",
                "api_key": "test-token",
            }
        ),
    )
    captured = _install_stub(monkeypatch)

    databricks_supervisor_executor._build_supervisor_executor()

    # Model threaded through verbatim — proves the env var → kwarg
    # mapping is correct. A regression that swapped the env-var
    # name (e.g. ``HARNESS_SUPERVISOR_MODEL_ID``) would make this
    # fail.
    assert captured.model == "databricks-claude-sonnet-4-6"
    # Explicit connection wins over profile resolution: base_url
    # and api_key match the env JSON exactly.
    assert captured.base_url == "https://example.databricks.com/ai-gateway/mlflow/v1"
    assert captured.api_key == "test-token"
    # No tools env var — supervisor_tools should be None.
    assert captured.supervisor_tools is None
    # Exactly one construction — the wrap's cache should not have
    # spawned a second runtime executor for the same call.
    assert captured.construction_count == 1


def test_executor_factory_decodes_tools_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``HARNESS_SUPERVISOR_TOOLS_JSON`` decodes into the nested list.

    AP-side serializes ``spec.executor.supervisor_tools`` via
    :func:`json.dumps`; the wrap must reconstruct the verbatim
    nested-shape list so the runtime executor forwards it to the
    gateway untouched. Asserts a round-trip on a real-shaped tool
    entry: type discriminator + nested config sub-dict + arbitrary
    string fields.
    """
    monkeypatch.setenv("HARNESS_SUPERVISOR_MODEL", "databricks-claude-sonnet-4-6")
    monkeypatch.setenv(
        "HARNESS_SUPERVISOR_CONNECTION_JSON",
        json.dumps(
            {
                "base_url": "https://example.databricks.com/ai-gateway/mlflow/v1",
                "api_key": "tok",
            }
        ),
    )
    tools = [
        {
            "type": "uc_connection",
            "uc_connection": {
                "name": "system_ai_agent_google_drive",
                "description": "Search Google Drive",
            },
        },
        {
            "type": "genie_space",
            "genie_space": {
                "id": "spaces/example",
                "description": "Sales Genie",
            },
        },
    ]
    monkeypatch.setenv("HARNESS_SUPERVISOR_TOOLS_JSON", json.dumps(tools))
    captured = _install_stub(monkeypatch)

    databricks_supervisor_executor._build_supervisor_executor()

    # The decoded list must match the original byte-for-byte. A
    # regression that lossy-translated the nested shape would
    # mutate the inner dict and the gateway would reject the
    # request with INVALID_PARAMETER_VALUE.
    assert captured.supervisor_tools == tools


def test_executor_factory_missing_model_raises_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No model env var → loud :class:`ValueError`.

    A spawned subprocess that booted without
    ``HARNESS_SUPERVISOR_MODEL`` would silently send empty-model
    requests to the gateway and 4xx; failing loud here makes the
    misconfiguration grep-discoverable from a single error message.
    """
    monkeypatch.delenv("HARNESS_SUPERVISOR_MODEL", raising=False)
    monkeypatch.delenv("HARNESS_SUPERVISOR_DATABRICKS_PROFILE", raising=False)
    monkeypatch.delenv("HARNESS_SUPERVISOR_TOOLS_JSON", raising=False)
    monkeypatch.delenv("HARNESS_SUPERVISOR_CONNECTION_JSON", raising=False)

    with pytest.raises(ValueError) as excinfo:
        databricks_supervisor_executor._build_supervisor_executor()

    # The message must name the specific env var so the operator
    # can fix the spawn-env builder without spelunking through
    # source.
    assert "HARNESS_SUPERVISOR_MODEL" in str(excinfo.value)


def test_executor_factory_partial_connection_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial connection JSON falls through to profile resolution.

    The wrap's contract is "both base_url AND api_key together, or
    neither." An object with only one of them is treated as if
    neither were set — credential resolution proceeds via the
    profile (or DEFAULT). This matches the runtime executor's
    own ``_resolve_gateway_credentials`` semantics.
    """
    monkeypatch.setenv("HARNESS_SUPERVISOR_MODEL", "databricks-claude-sonnet-4-6")
    monkeypatch.setenv(
        "HARNESS_SUPERVISOR_CONNECTION_JSON",
        # Only base_url; missing api_key. The wrap must NOT
        # construct the executor with a half-populated
        # connection, because the runtime executor would then
        # send unauthenticated requests.
        json.dumps({"base_url": "https://example.databricks.com/ai-gateway/mlflow/v1"}),
    )
    captured = _install_stub(monkeypatch)

    def _fake_resolve(profile: str | None) -> Any:
        # The fall-through path was reached — record on the
        # shared captured record so the assertions stay simple.
        captured.profile = profile

        class _StubCreds:
            host = "https://resolved.databricks.com"
            token = "resolved-token"

        return _StubCreds()

    monkeypatch.setattr(
        "omnigent.runtime.credentials.databricks.resolve_databricks_workspace",
        _fake_resolve,
    )

    databricks_supervisor_executor._build_supervisor_executor()

    # Profile resolver was invoked (with None — no
    # HARNESS_SUPERVISOR_DATABRICKS_PROFILE set), proving the
    # half-populated connection was correctly discarded.
    assert captured.profile is None
    # The runtime executor was constructed with the resolved
    # creds, not the half-populated explicit ones. base_url
    # comes from the resolver's host + GATEWAY_PATH composition.
    assert captured.base_url == ("https://resolved.databricks.com/ai-gateway/mlflow/v1")
    assert captured.api_key == "resolved-token"


# ── _resolve_env_json: malformed parent-AP env vars must raise ────


@pytest.mark.parametrize(
    "env_var",
    [
        "HARNESS_SUPERVISOR_TOOLS_JSON",
        "HARNESS_SUPERVISOR_CONNECTION_JSON",
    ],
)
def test_executor_factory_raises_on_malformed_env_json(
    env_var: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A non-JSON value in either ``HARNESS_SUPERVISOR_TOOLS_JSON`` or
    ``HARNESS_SUPERVISOR_CONNECTION_JSON`` must RAISE rather than
    silently fall through to ``None``. These env vars are written by
    the parent omnigent spawn path; a malformed value is parent-side
    misbehavior, not a user-recoverable input. Silent fallback would
    drop, e.g., an explicit connection override and route requests at
    a different workspace than the caller asked for.
    """
    monkeypatch.setenv("HARNESS_SUPERVISOR_MODEL", "databricks-claude-sonnet-4-6")
    monkeypatch.setenv(env_var, "{not-json")
    _install_stub(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        databricks_supervisor_executor._build_supervisor_executor()

    assert env_var in str(excinfo.value)


@pytest.mark.parametrize(
    "env_var,bad_payload,expected_type_name",
    [
        # TOOLS_JSON expects list — dict here is wrong-shape.
        ("HARNESS_SUPERVISOR_TOOLS_JSON", '{"k": 1}', "list"),
        # CONNECTION_JSON expects dict — list here is wrong-shape.
        ("HARNESS_SUPERVISOR_CONNECTION_JSON", "[1, 2]", "dict"),
    ],
)
def test_executor_factory_raises_on_wrong_shape_env_json(
    env_var: str,
    bad_payload: str,
    expected_type_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Valid JSON of the WRONG shape (e.g. a dict where a list is
    required) must also raise. Without this guard, a parent that
    accidentally writes the wrong type drops the value silently.
    """
    monkeypatch.setenv("HARNESS_SUPERVISOR_MODEL", "databricks-claude-sonnet-4-6")
    monkeypatch.setenv(env_var, bad_payload)
    _install_stub(monkeypatch)

    with pytest.raises(ValueError) as excinfo:
        databricks_supervisor_executor._build_supervisor_executor()

    msg = str(excinfo.value)
    assert env_var in msg
    assert expected_type_name in msg


# ── _translate_event: runtime → inner event mapping ───────────────


def test_translate_event_text_chunk_passthrough() -> None:
    """
    Runtime ``TextChunk`` maps 1:1 to inner ``TextChunk`` carrying
    the same text. Trivial branch but kept here so a regression in
    the dispatch order (e.g. catching ``object`` first) shows up.
    """
    out = databricks_supervisor_executor._translate_event(RuntimeTextChunk(text="hello"))

    assert len(out) == 1
    assert isinstance(out[0], InnerTextChunk)
    assert out[0].text == "hello"


def test_translate_event_tool_call_observed_fans_out() -> None:
    """
    A single runtime ``ToolCallObserved`` produces TWO inner events
    (a ``ToolCallRequest`` followed by a ``ToolCallComplete``) so
    the adapter can emit them as a function_call + function_call_output
    pair. Omnigent re-pairs back into a single ``ToolCallObserved``.
    Both events MUST carry the same ``call_id`` in their metadata
    so the AP-side correlator works.
    """
    out = databricks_supervisor_executor._translate_event(
        RuntimeToolCallObserved(
            call_id="call_xyz",
            name="list_files",
            arguments={"q": "*"},
            result="a, b, c",
            status="success",
            duration_ms=42.0,
        )
    )

    assert len(out) == 2
    assert isinstance(out[0], ToolCallRequest)
    assert out[0].name == "list_files"
    assert out[0].args == {"q": "*"}
    assert out[0].metadata == {"call_id": "call_xyz"}

    assert isinstance(out[1], ToolCallComplete)
    assert out[1].name == "list_files"
    assert out[1].status == ToolCallStatus.SUCCESS
    assert out[1].result == "a, b, c"
    assert out[1].duration_ms == 42.0
    assert out[1].metadata == {"call_id": "call_xyz"}


def test_translate_event_tool_call_unknown_status_falls_back_to_error() -> None:
    """
    When the runtime status string isn't one of the inner
    ``ToolCallStatus`` enum values, the translator must fall back
    to ``ERROR`` rather than letting a ``KeyError`` escape — and
    must NOT silently coerce to ``SUCCESS``. Guards against a
    regression that swaps the dict (e.g. defaulting to SUCCESS).
    """
    out = databricks_supervisor_executor._translate_event(
        RuntimeToolCallObserved(
            call_id="call_unknown",
            name="t",
            arguments={},
            result="",
            status="some-future-status-we-do-not-know",
            duration_ms=0.0,
        )
    )

    completes = [e for e in out if isinstance(e, ToolCallComplete)]
    assert len(completes) == 1
    assert completes[0].status is ToolCallStatus.ERROR


def test_translate_event_executor_error_folds_code_into_prefix() -> None:
    """
    Inner ``ExecutorError`` has no separate ``code`` field, so the
    translator folds the runtime ``code`` into the message prefix
    as ``[code] message``. The ``retryable`` flag passes through
    verbatim.
    """
    out = databricks_supervisor_executor._translate_event(
        RuntimeExecutorError(
            message="boom",
            code="auth_failed",
            retryable=True,
        )
    )

    assert len(out) == 1
    assert isinstance(out[0], InnerExecutorError)
    assert out[0].message == "[auth_failed] boom"
    assert out[0].retryable is True


def test_translate_event_executor_error_without_code_no_prefix() -> None:
    """
    When the runtime error has no ``code``, no ``[...] `` prefix is
    prepended — guards against a regression that always prefixes
    ``[None] `` or ``[] ``.
    """
    out = databricks_supervisor_executor._translate_event(
        RuntimeExecutorError(message="boom", retryable=False)
    )

    assert len(out) == 1
    assert isinstance(out[0], InnerExecutorError)
    assert out[0].message == "boom"
    assert out[0].retryable is False


def test_translate_event_turn_complete_drops_runtime_metadata() -> None:
    """
    Runtime ``TurnComplete`` carries ``usage`` / ``response_model`` /
    ``response_id`` / ``finish_reasons`` that the inner event has no
    field for. The translator must surface only ``response`` (the
    text). This documents the known event-fidelity gap so a future
    inner-event extension forces this test to update.
    """
    out = databricks_supervisor_executor._translate_event(
        RuntimeTurnComplete(
            text="all done",
            usage={"input_tokens": 100, "output_tokens": 50},
            response_model="claude",
            response_id="resp_x",
            finish_reasons=["stop"],
        )
    )

    assert len(out) == 1
    assert isinstance(out[0], InnerTurnComplete)
    assert out[0].response == "all done"


def test_translate_event_unknown_runtime_event_warns_and_drops(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    The translator's ``object`` arm exists so a future runtime
    event type doesn't tear the harness down. Unknown events must
    produce zero inner events AND log a warning so the gap is
    visible in production rather than silently swallowed.
    """

    class _NewRuntimeEvent:
        pass

    with caplog.at_level(logging.WARNING):
        out = databricks_supervisor_executor._translate_event(_NewRuntimeEvent())

    assert out == []
    assert any(
        "_NewRuntimeEvent" in rec.getMessage() and "dropping" in rec.getMessage()
        for rec in caplog.records
    )
