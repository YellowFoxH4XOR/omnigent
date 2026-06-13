"""
Tests for the built-in cost-budget policy
(:mod:`omnigent.policies.builtins.cost`) — the ``cost_budget`` factory.

The policy gates the ``tool_call`` phase: ASK at each soft warning
checkpoint, and once the hard limit is reached DENY tool calls while the
session is still on an expensive model (forcing a ``/model`` downgrade),
ALLOW once it has switched to a cheaper one.

Layers:

- **Layer 1** — direct callable on the ``tool_call`` phase: ALLOW below
  the soft checkpoints, ASK (recorded via ``session_state`` so an
  approved checkpoint doesn't re-prompt) when one is crossed, DENY over
  the hard limit on an expensive/unknown model, ALLOW over the limit on
  a cheaper model, abstain on every other phase, and factory validation.
- **Layer 2** — spec resolution through :func:`resolve_function_policy`,
  proving DENY and ASK thread through the engine boundary with the cost
  on ``EvaluationContext.usage`` and the active model on
  ``EvaluationContext.model``.
- **Layer 3** — registry discovery: the one ``POLICY_REGISTRY`` factory
  entry is browsable and its schema validates good / bad params.
"""

from __future__ import annotations

from typing import Any

import pytest

from omnigent.policies.builtins.cost import _ASK_APPROVED_KEY, cost_budget
from omnigent.policies.function import FunctionPolicy, resolve_function_policy
from omnigent.policies.registry import get_registry, load_registry, validate_factory_params
from omnigent.policies.schema import PolicyEvent
from omnigent.policies.types import EvaluationContext
from omnigent.spec.types import FunctionPolicySpec, FunctionRef, Phase, PolicyAction

_HANDLER = "omnigent.policies.builtins.cost.cost_budget"


def _tool(
    cost: float | None,
    *,
    model: str | None = "databricks-claude-opus-4-8",
    session_state: dict[str, Any] | None = None,
    harness: str | None = None,
) -> PolicyEvent:
    """
    Build a ``tool_call`` :class:`PolicyEvent` with a cost + active model.

    :param cost: ``total_cost_usd`` to place under ``context.usage``,
        e.g. ``2.5``. ``None`` omits the field entirely (the
        unpriced-session case).
    :param model: Active model under ``context.model``, e.g.
        ``"databricks-claude-opus-4-8"`` or the tier alias ``"opus"``.
        Defaults to an expensive (Opus) model; pass ``None`` for the
        undeterminable-model case.
    :param session_state: Optional persisted state, e.g.
        ``{_ASK_APPROVED_KEY: 2.0}``. ``None`` means empty.
    :param harness: Harness under ``context.harness``, e.g.
        ``"codex-native"`` (a native hook stamped it). ``None`` is the
        web / API / unstamped case, where the deny message stays
        surface-agnostic.
    :returns: A ``tool_call`` event dict.
    """
    usage: dict[str, Any] = {} if cost is None else {"total_cost_usd": cost}
    return {
        "type": "tool_call",
        "target": "sys_os_shell",
        "data": {"name": "sys_os_shell", "arguments": {}},
        "context": {"actor": {}, "usage": usage, "model": model, "harness": harness},
        "session_state": session_state or {},
    }


def _event(phase: str, cost: float) -> PolicyEvent:
    """
    Build a non-tool-call :class:`PolicyEvent` carrying a session cost.

    :param phase: Event type, e.g. ``"request"`` / ``"response"`` /
        ``"tool_result"``.
    :param cost: ``total_cost_usd`` under ``context.usage``, e.g. ``9.99``.
    :returns: An event dict of the given phase (over budget, to prove the
        non-tool-call phases are not gated).
    """
    return {
        "type": phase,
        "target": None,
        "data": "x",
        "context": {"actor": {}, "usage": {"total_cost_usd": cost}, "model": "opus"},
        "session_state": {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — direct callable
# ══════════════════════════════════════════════════════════════════════════════


def test_below_ask_threshold_allows() -> None:
    """Spend under the lowest checkpoint abstains (ALLOW)."""
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0])
    assert policy(_tool(1.0)) == {"result": "ALLOW"}


def test_crossing_a_checkpoint_asks_and_records_it() -> None:
    """Crossing a checkpoint (unapproved) → ASK + record the crossed value.

    The ASK must carry a ``state_updates`` SET recording the crossed
    checkpoint so it (and lower ones) don't re-prompt once approved. A
    missing ``state_updates`` would mean the user is asked on every
    subsequent tool call even after approving.
    """
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0, 4.0])
    result = policy(_tool(2.0))  # exactly at the first checkpoint — `>=`
    assert result["result"] == "ASK"
    # SET highwater = 2.0: applied on approve so $2 (and lower) stop prompting.
    assert result["state_updates"] == [
        {"key": _ASK_APPROVED_KEY, "action": "set", "value": 2.0},
    ]


def test_approved_checkpoint_does_not_reprompt_higher_one_does() -> None:
    """Approved $2 → a $3 tool call is silent; reaching $4 ASKs again.

    Proves the "ASK at several amounts, once each on approve" behavior:
    the recorded highwater suppresses lower checkpoints, the next higher
    checkpoint still fires.
    """
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0, 4.0])
    # Already approved past $2 → a $3 tool call is allowed (no re-prompt).
    assert policy(_tool(3.0, session_state={_ASK_APPROVED_KEY: 2.0})) == {"result": "ALLOW"}
    # Crossing the next checkpoint ($4) prompts again.
    result = policy(_tool(4.0, session_state={_ASK_APPROVED_KEY: 2.0}))
    assert result["result"] == "ASK"
    assert result["state_updates"] == [
        {"key": _ASK_APPROVED_KEY, "action": "set", "value": 4.0},
    ]


def test_declined_checkpoint_reasks_until_approved() -> None:
    """A checkpoint not yet recorded re-asks on every tool call.

    A decline never writes the highwater (the engine withholds an ASK's
    ``state_updates`` on decline), so the next tool call still over the
    same threshold must ASK again — the gate keeps blocking until the
    user approves, not just once. Calling the policy twice with the same
    un-recorded state must ASK both times.
    """
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0])
    first = policy(_tool(3.0, session_state={}))
    second = policy(_tool(3.0, session_state={}))
    assert first["result"] == "ASK"
    assert second["result"] == "ASK"  # not recorded → re-asks


def test_over_budget_on_expensive_model_denies() -> None:
    """Over the hard limit on an expensive model → DENY (force downgrade).

    The default expensive set includes Opus; an over-budget tool call on
    Opus must be blocked, and the reason must surface the spend figure and
    the high-cost model tokens so the user knows what to avoid. If this
    ALLOWed, the budget would never bite on the costly model it targets.
    """
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0])
    result = policy(_tool(6.0, model="databricks-claude-opus-4-8"))
    assert result["result"] == "DENY"
    assert "6.00" in result["reason"]  # current cost surfaced
    # The high-cost tokens are listed so the user knows which to avoid.
    assert "opus" in result["reason"]
    assert "gpt-5.5" in result["reason"]


def test_deny_reason_for_codex_points_to_terminal() -> None:
    """A codex-native session's deny reason says to switch in the terminal.

    Codex has no web model picker — the only way to switch is the terminal
    TUI's ``/model`` — so the verbatim instruction must name both. If this
    regressed to the surface-agnostic wording, a codex user would not be
    told the one mechanism that actually works for them.
    """
    policy = cost_budget(max_cost_usd=5.0)
    result = policy(_tool(6.0, model="opus", harness="codex-native"))
    assert result["result"] == "DENY"
    assert "in the terminal" in result["reason"]
    assert "/model" in result["reason"]


def test_deny_reason_for_non_codex_omits_terminal() -> None:
    """A non-codex (or unstamped) session's deny reason stays surface-agnostic.

    Claude / web / API sessions are not terminal-only (they have a model
    picker), so the message must NOT tell them to use the terminal or
    ``/model`` — it would be wrong/confusing. This is the regression guard
    for "only codex says in the terminal".
    """
    policy = cost_budget(max_cost_usd=5.0)
    # harness=None mirrors the web/API path (no native hook stamped it).
    result = policy(_tool(6.0, model="opus", harness=None))
    assert result["result"] == "DENY"
    assert "in the terminal" not in result["reason"]
    assert "/model" not in result["reason"]
    assert "switch to a cheaper model" in result["reason"]


def test_over_budget_on_cheaper_model_allows() -> None:
    """Over the hard limit on a cheaper model → ALLOW (downgrade satisfied).

    Once the session has switched off an expensive model, the budget
    becomes a no-op — the whole point of a "downgrade gate" rather than a
    hard stop. A DENY here would trap the user even after they complied.
    """
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0])
    assert policy(_tool(6.0, model="claude-sonnet-4-6")) == {"result": "ALLOW"}


def test_over_budget_unknown_model_denies_fail_closed() -> None:
    """Over the hard limit with no determinable model → DENY (fail closed).

    When the engine could not resolve a model (``None``), the gate cannot
    confirm a cheaper model, so it blocks and asks the user to pick one
    with ``/model`` rather than silently allowing unbounded spend. ALLOW
    here would let an over-budget session run unchecked whenever the model
    is unknown.
    """
    policy = cost_budget(max_cost_usd=5.0)
    assert policy(_tool(6.0, model=None))["result"] == "DENY"


def test_hard_limit_wins_over_checkpoint_approval() -> None:
    """Over the hard limit on an expensive model → DENY even if approved.

    A prior checkpoint approval must not let an over-budget session keep
    calling tools on the costly model; the hard gate is checked before
    the soft checkpoints.
    """
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0, 4.0])
    result = policy(_tool(5.0, model="opus", session_state={_ASK_APPROVED_KEY: 4.0}))
    assert result["result"] == "DENY"


def test_default_expensive_set_matches_opus_and_gpt55() -> None:
    """The default expensive set blocks Fable, Opus, and GPT-5.5 spellings.

    Substring + case-insensitive matching must hit the deployment ids in
    this stack: ``databricks-claude-opus-4-8`` and ``databricks-gpt-5-5``
    (dashes). A miss would let the costliest models run past budget under
    the zero-config default.
    """
    policy = cost_budget(max_cost_usd=5.0)
    assert policy(_tool(6.0, model="databricks-claude-opus-4-8"))["result"] == "DENY"
    assert policy(_tool(6.0, model="databricks-gpt-5-5"))["result"] == "DENY"
    # Fable is the costliest tier (above Opus at 2x its price); both the
    # concrete id and the bare picker alias must be gated, or switching to
    # Fable becomes a budget bypass for a session downgraded off Opus.
    assert policy(_tool(6.0, model="claude-fable-5"))["result"] == "DENY"
    assert policy(_tool(6.0, model="fable"))["result"] == "DENY"
    # A non-listed model is treated as cheap → allowed over budget.
    assert policy(_tool(6.0, model="databricks-claude-haiku-4-5")) == {"result": "ALLOW"}


def test_custom_expensive_models_substring_case_insensitive() -> None:
    """A custom token matches case-insensitively as a substring.

    Proves the author can override the default set; ``"foo"`` must match
    ``"x-FOO-bar"`` so authors don't have to spell full provider-prefixed
    ids, and a non-matching model is allowed over budget.
    """
    policy = cost_budget(max_cost_usd=5.0, expensive_models=["FoO"])
    assert policy(_tool(6.0, model="x-foo-bar"))["result"] == "DENY"
    assert policy(_tool(6.0, model="claude-sonnet-4-6")) == {"result": "ALLOW"}


def test_empty_expensive_models_disables_hard_gate() -> None:
    """``expensive_models=[]`` disables the hard gate (soft thresholds only).

    Over budget on Opus must NOT be hard-DENYed (no model is blocked);
    with the soft checkpoint already approved it ALLOWs. The soft ASK
    still fires below the limit — proving the empty list scopes off only
    the hard block, not the whole policy.
    """
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0], expensive_models=[])
    # Over budget on Opus, checkpoint already approved → ALLOW (no hard DENY).
    assert policy(_tool(6.0, model="opus", session_state={_ASK_APPROVED_KEY: 2.0})) == {
        "result": "ALLOW"
    }
    # Soft checkpoint still asks below the limit.
    assert policy(_tool(2.0, model="opus"))["result"] == "ASK"


def test_abstains_on_non_tool_call_phases() -> None:
    """Only ``tool_call`` is gated — request/response/tool_result abstain.

    The cost gate runs at ``tool_call`` (the one phase a PreToolUse hook
    can block); an over-budget event of any other phase must ALLOW so the
    policy does not block text-only turns or post-hoc results.
    """
    policy = cost_budget(max_cost_usd=1.0, ask_thresholds_usd=[0.5])
    assert policy(_event("request", 9.99)) == {"result": "ALLOW"}
    assert policy(_event("response", 9.99)) == {"result": "ALLOW"}
    assert policy(_event("tool_result", 9.99)) == {"result": "ALLOW"}


def test_unpriced_session_never_trips() -> None:
    """No ``total_cost_usd`` (pricing unavailable) → ALLOW, never blocks.

    Defaults to ``0.0``; the policy cannot budget what it cannot price,
    so it must abstain rather than block every tool call at $0 — even on
    an expensive model.
    """
    policy = cost_budget(max_cost_usd=5.0, ask_thresholds_usd=[2.0])
    assert policy(_tool(None, model="opus")) == {"result": "ALLOW"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cost_usd": 0.0},  # non-positive hard limit
        {"max_cost_usd": -1.0},  # negative hard limit
        {"max_cost_usd": 5.0, "ask_thresholds_usd": [5.0]},  # not strictly below max
        {"max_cost_usd": 5.0, "ask_thresholds_usd": [6.0]},  # above max
        {"max_cost_usd": 5.0, "ask_thresholds_usd": [0.0]},  # not positive
        {"max_cost_usd": 5.0, "ask_thresholds_usd": [1.0, 6.0]},  # one above max
        {"max_cost_usd": 5.0, "expensive_models": [""]},  # empty model token
        {"max_cost_usd": 5.0, "expensive_models": [123]},  # non-string token
    ],
)
def test_factory_rejects_invalid_config(kwargs: dict[str, Any]) -> None:
    """Bad config fails loud at factory time (ValueError), not silently.

    A non-positive limit, a checkpoint outside ``(0, max_cost_usd)``, or a
    non-string / empty ``expensive_models`` entry is a misconfiguration
    that could never enforce correctly, so it must raise rather than build
    a dead gate.
    """
    with pytest.raises(ValueError):
        cost_budget(**kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — spec resolution through resolve_function_policy
# ══════════════════════════════════════════════════════════════════════════════


def _tool_ctx(cost: float, model: str | None) -> EvaluationContext:
    """
    Build a TOOL_CALL :class:`EvaluationContext` with cost + model set.

    Mirrors what the engine injects (``usage`` + ``model``) so a directly
    resolved policy sees the same ``event["context"]`` it would in
    production.

    :param cost: ``total_cost_usd`` for the usage context, e.g. ``6.0``.
    :param model: Active model id for ``ctx.model``, e.g. ``"opus"`` or
        ``None``.
    :returns: A ready-to-evaluate TOOL_CALL context.
    """
    return EvaluationContext(
        phase=Phase.TOOL_CALL,
        content={"name": "sys_os_shell", "arguments": {}},
        tool_name="sys_os_shell",
        usage={"total_cost_usd": cost},
        model=model,
    )


@pytest.mark.asyncio
async def test_resolve_from_spec_denies_over_budget_on_expensive_model() -> None:
    """Over-budget on an expensive model DENYs through the engine boundary.

    Proves the cost on ``EvaluationContext.usage`` AND the model on
    ``EvaluationContext.model`` both reach the resolved callable (via
    ``event["context"]``) and the DENY threads back as a
    :class:`PolicyAction`.
    """
    spec = FunctionPolicySpec(
        name="cost",
        on=None,
        function=FunctionRef(path=_HANDLER, arguments={"max_cost_usd": 5.0}),
    )
    policy: FunctionPolicy = resolve_function_policy(spec)
    result = await policy.evaluate(_tool_ctx(6.0, "databricks-claude-opus-4-8"), {})
    assert result.action == PolicyAction.DENY


@pytest.mark.asyncio
async def test_resolve_from_spec_allows_over_budget_on_cheaper_model() -> None:
    """Over-budget on a cheaper model ALLOWs through the engine boundary.

    The model on ``EvaluationContext.model`` must let a downgraded session
    through — proving the model gate (not just the cost) crosses the
    boundary.
    """
    spec = FunctionPolicySpec(
        name="cost",
        on=None,
        function=FunctionRef(path=_HANDLER, arguments={"max_cost_usd": 5.0}),
    )
    policy: FunctionPolicy = resolve_function_policy(spec)
    result = await policy.evaluate(_tool_ctx(6.0, "claude-sonnet-4-6"), {})
    assert result.action == PolicyAction.ALLOW


@pytest.mark.asyncio
async def test_resolve_from_spec_asks_in_soft_zone() -> None:
    """Soft-zone spend surfaces as ASK through the engine boundary."""
    spec = FunctionPolicySpec(
        name="cost",
        on=None,
        function=FunctionRef(
            path=_HANDLER, arguments={"max_cost_usd": 5.0, "ask_thresholds_usd": [2.0]}
        ),
    )
    policy: FunctionPolicy = resolve_function_policy(spec)
    result = await policy.evaluate(_tool_ctx(3.0, "opus"), {})
    assert result.action == PolicyAction.ASK


# ══════════════════════════════════════════════════════════════════════════════
# Layer 3 — registry discovery
# ══════════════════════════════════════════════════════════════════════════════


def test_registry_discovers_cost_budget() -> None:
    """The cost_budget factory is browsable in the policy registry."""
    load_registry()
    by_handler = {e.handler: e for e in get_registry()}
    assert _HANDLER in by_handler
    assert by_handler[_HANDLER].kind == "factory"
    assert by_handler[_HANDLER].params_schema is not None


def test_registry_validates_factory_params() -> None:
    """The registry schema accepts good params and rejects bad ones."""
    load_registry()
    # Valid: required hard limit alone, with the soft gate, and with models.
    assert validate_factory_params(_HANDLER, {"max_cost_usd": 5.0}) is None
    assert (
        validate_factory_params(_HANDLER, {"max_cost_usd": 5.0, "ask_thresholds_usd": [2.0]})
        is None
    )
    assert (
        validate_factory_params(_HANDLER, {"max_cost_usd": 5.0, "expensive_models": ["opus"]})
        is None
    )
    # Wrong type for the checkpoints (must be an array, not a scalar).
    assert (
        validate_factory_params(_HANDLER, {"max_cost_usd": 5.0, "ask_thresholds_usd": 2.0})
        is not None
    )
    # Wrong type for expensive_models (must be an array, not a scalar).
    assert (
        validate_factory_params(_HANDLER, {"max_cost_usd": 5.0, "expensive_models": "opus"})
        is not None
    )
    # Missing the required hard limit.
    assert validate_factory_params(_HANDLER, {}) is not None
    # Unknown param.
    err_unknown = validate_factory_params(_HANDLER, {"max_cost_usd": 5.0, "bogus": 1})
    assert err_unknown is not None and "bogus" in err_unknown
    # Wrong type for the hard limit.
    assert validate_factory_params(_HANDLER, {"max_cost_usd": "lots"}) is not None
