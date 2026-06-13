"""Tests for server-side policy evaluation (steps 5.5 and 5.6).

Verifies that ``POST /v1/sessions/{id}/events`` evaluates tool
calls (``function_call`` with ``evaluate_policy: true``) and
user input (``message`` with ``role: "user"``) against the agent
spec's guardrails and returns the correct verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from omnigent.entities import Conversation, ConversationItem
from omnigent.entities.agent import Agent, LoadedAgent
from omnigent.entities.conversation import FunctionCallData
from omnigent.policies.types import PolicyAction, PolicyResult
from omnigent.server.routes.sessions import (
    _build_skill_slash_command_policy_body,
    _evaluate_input_policy,
    _evaluate_tool_call_policy,
)
from omnigent.server.schemas import SessionEventInput
from omnigent.spec import AgentSpec
from omnigent.spec.types import PolicySpec

# ── Stub stores ──────────────────────────────────────────────


@dataclass
class _FakeConversationStore:
    """Minimal conversation store for policy evaluation tests.

    :param labels: Pre-seeded labels returned by
        ``get_conversation_labels``.
    :param appended_items: Items captured by ``append`` calls.
    """

    labels: dict[str, str] = field(default_factory=dict)
    appended_items: list[Any] = field(default_factory=list)

    def get_conversation(self, conversation_id: str) -> Conversation:
        """Return a stub conversation.

        :param conversation_id: Session id, e.g. ``"sess_1"``.
        :returns: Stub conversation with agent_id set.
        """
        return Conversation(
            id=conversation_id,
            created_at=1,
            updated_at=1,
            root_conversation_id=conversation_id,
            agent_id="ag_test",
        )

    def get_conversation_labels(self, conversation_id: str) -> dict[str, str]:
        """Return pre-seeded labels.

        :param conversation_id: Session id.
        :returns: Label dict.
        """
        return dict(self.labels)

    def set_labels(self, conversation_id: str, labels: dict[str, str]) -> None:
        """Record label writes.

        :param conversation_id: Session id.
        :param labels: Labels to write.
        """
        self.labels.update(labels)

    def append(self, conversation_id: str, items: list[Any]) -> list[ConversationItem]:
        """Record appended items and return stubs.

        :param conversation_id: Session id.
        :param items: Items to persist.
        :returns: List of stub conversation items with generated ids.
        """
        result = []
        for i, item in enumerate(items):
            ci = ConversationItem(
                id=f"item_{i}",
                type=getattr(item, "type", "function_call"),
                data=FunctionCallData(
                    agent="test-agent",
                    name="sys_os_shell",
                    arguments="{}",
                    call_id="call_1",
                ),
                created_at=1,
                status="completed",
            )
            result.append(ci)
            self.appended_items.append(item)
        return result


@dataclass
class _FakeAgentStore:
    """Minimal agent store that returns a stub agent.

    :param agent: The agent to return from ``get()``.
    """

    agent: Agent | None = None

    def get(self, agent_id: str) -> Agent | None:
        """Return the pre-configured agent.

        :param agent_id: Agent id.
        :returns: The stub agent or None.
        """
        return self.agent


@dataclass
class _FakeBody:
    """Minimal SessionEventInput stub.

    :param type: Event type, e.g. ``"function_call"``.
    :param data: Event data dict.
    """

    type: str
    data: dict[str, Any]


# ── Helpers ──────────────────────────────────────────────────


def _make_function_call_body(
    name: str = "sys_os_shell",
    arguments: str = '{"command": "ls"}',
    call_id: str = "call_1",
) -> _FakeBody:
    """Build a function_call event body with evaluate_policy.

    :param name: Tool name, e.g. ``"sys_os_shell"``.
    :param arguments: JSON-encoded arguments string.
    :param call_id: Call identifier.
    :returns: A fake body matching SessionEventInput shape.
    """
    return _FakeBody(
        type="function_call",
        data={
            "name": name,
            "arguments": arguments,
            "call_id": call_id,
            "model": "test-agent",
            "evaluate_policy": True,
        },
    )


def _make_agent(agent_id: str = "ag_test") -> Agent:
    """Build a stub Agent entity.

    :param agent_id: Agent identifier.
    :returns: Agent with minimal fields.
    """
    return Agent(
        id=agent_id,
        created_at=1,
        name="test-agent",
        bundle_location="ag_test/abc123",
    )


def _make_spec_no_guardrails() -> AgentSpec:
    """Build an AgentSpec with no guardrails.

    :returns: Minimal AgentSpec with guardrails=None.
    """
    return AgentSpec(spec_version=1, name="test-agent")


# ── Tests ────────────────────────────────────────────────────


_CACHE_PATCH = "omnigent.server.routes.sessions.get_agent_cache"
_ENGINE_PATCH = "omnigent.server.routes.sessions.build_policy_engine"
_STREAM_PATCH = "omnigent.server.routes.sessions.session_stream"


@pytest.mark.asyncio
async def test_allow_verdict():
    """Policy evaluation returns allow when the engine ALLOWs."""
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_function_call_body()

    spec = _make_spec_no_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    allow_result = PolicyResult(action=PolicyAction.ALLOW)

    async def _eval(_ctx: Any) -> PolicyResult:
        return allow_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        mock_engine.apply_label_writes = lambda x: None

        result = await _evaluate_tool_call_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result is None


@pytest.mark.asyncio
async def test_deny_verdict():
    """Policy evaluation returns deny with reason when the
    engine DENYs.
    """
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_function_call_body()

    spec = _make_spec_no_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    deny_result = PolicyResult(
        action=PolicyAction.DENY,
        reason="Tool blocked by policy",
    )

    async def _eval(_ctx: Any) -> PolicyResult:
        return deny_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        mock_engine.apply_label_writes = lambda x: None

        result = await _evaluate_tool_call_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result["verdict"] == "deny"
    assert result["reason"] == "Tool blocked by policy"


@pytest.mark.asyncio
async def test_pending_verdict_registers_elicitation():
    """Policy evaluation returns pending and registers an
    elicitation when the engine returns ASK.
    """
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_function_call_body()

    spec = _make_spec_no_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    ask_result = PolicyResult(
        action=PolicyAction.ASK,
        reason="Requires user approval",
        deciding_policy="approve_shell",
    )

    async def _eval(_ctx: Any) -> PolicyResult:
        return ask_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
        patch(_STREAM_PATCH),
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        # No per-policy override → the spec-wide engine value applies.
        mock_engine.spec_for = lambda _name: None
        mock_engine.ask_timeout = 30

        result = await _evaluate_tool_call_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result["verdict"] == "pending"
    assert "elicitation_id" in result
    assert result["elicitation_id"].startswith("elicit_")
    # The pending verdict carries the spec-resolved approval window so
    # the runner's park honors it; without it the runner falls back to
    # its hard-coded 120s default regardless of the spec.
    assert result["ask_timeout"] == 30
    # Approval state lives on the runner (in-memory dict), not
    # the task store. The server just publishes the SSE event.
    # No pending_tool_call row is created.


@pytest.mark.asyncio
async def test_pending_verdict_carries_per_policy_ask_timeout():
    """The deciding policy's ``ask_timeout`` override rides the verdict.

    A spec that grants one expensive ASK a longer window (e.g. nessie's
    pi worker setting a day-long approval) must reach the runner's park;
    if the verdict carried the spec-wide value instead, the override
    would be silently ignored on every runner-dispatched tool call.
    """
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_function_call_body()

    spec = _make_spec_no_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    ask_result = PolicyResult(
        action=PolicyAction.ASK,
        reason="Requires user approval",
        deciding_policy="approve_shell",
    )

    async def _eval(_ctx: Any) -> PolicyResult:
        return ask_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
        patch(_STREAM_PATCH),
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        # Real PolicySpec: the resolver reads ``.ask_timeout`` off the
        # deciding policy's spec, overriding the engine-wide 30.
        mock_engine.spec_for = lambda _name: PolicySpec(
            name="approve_shell", on=None, ask_timeout=86400
        )
        mock_engine.ask_timeout = 30

        result = await _evaluate_tool_call_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result["verdict"] == "pending"
    # 86400 (per-policy) — not 30 (engine-wide): the override wins.
    assert result["ask_timeout"] == 86400


@pytest.mark.asyncio
async def test_no_agent_binding_skips_policy():
    """When the session has no agent_id, policy evaluation is
    skipped and the function_call is persisted with allow verdict.
    """
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=None)
    conv = Conversation(
        id="sess_1",
        created_at=1,
        updated_at=1,
        root_conversation_id="sess_1",
        agent_id=None,
    )
    body = _make_function_call_body()

    result = await _evaluate_tool_call_policy(
        "sess_1",
        conv,
        body,
        conv_store,
        agent_store,
        None,
    )

    assert result is None


# ── INPUT policy tests (step 5.6) ───────────────────────────


def _make_user_message_body(
    text: str = "hello tell me about canada",
) -> _FakeBody:
    """Build a user message event body.

    :param text: User message text.
    :returns: A fake body matching SessionEventInput shape
        for a user message.
    """
    return _FakeBody(
        type="message",
        data={
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        },
    )


def _make_spec_with_guardrails() -> AgentSpec:
    """Build an AgentSpec with a guardrails block (but no policies).

    The presence of ``guardrails`` triggers policy engine
    construction; the empty policy list means ALLOW by default.

    :returns: AgentSpec with guardrails enabled.
    """
    from omnigent.spec.types import GuardrailsSpec

    return AgentSpec(
        spec_version=1,
        name="test-agent",
        guardrails=GuardrailsSpec(),
    )


@pytest.mark.asyncio
async def test_input_allow_verdict():
    """INPUT policy evaluation returns allow when the engine ALLOWs."""
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_user_message_body()

    spec = _make_spec_with_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    allow_result = PolicyResult(action=PolicyAction.ALLOW)

    async def _eval(_ctx: Any) -> PolicyResult:
        return allow_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        mock_engine.apply_label_writes = lambda x: None

        result = await _evaluate_input_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result is None


@pytest.mark.asyncio
async def test_input_deny_verdict():
    """INPUT policy evaluation returns deny when the engine DENYs."""
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_user_message_body("hello tell me about canada")

    spec = _make_spec_with_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    deny_result = PolicyResult(
        action=PolicyAction.DENY,
        reason="Input mentions Canada",
    )

    async def _eval(_ctx: Any) -> PolicyResult:
        return deny_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        mock_engine.apply_label_writes = lambda x: None

        result = await _evaluate_input_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result["verdict"] == "deny"
    assert result["reason"] == "Input mentions Canada"


@pytest.mark.asyncio
async def test_skill_slash_command_policy_body_uses_typed_command_text():
    """
    Skill slash-command input policy evaluates typed user text.

    The policy surface must be ``/<skill> <arguments>``, not the
    hidden meta message that contains the full skill instructions.
    Otherwise a bundled skill body could trip input guardrails before
    the user has made a request.
    """
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    slash_body = SessionEventInput(
        type="slash_command",
        data={
            "kind": "skill",
            "name": "grill-me",
            "arguments": "review Canada rollout",
        },
    )
    policy_body = _build_skill_slash_command_policy_body(slash_body)

    spec = _make_spec_with_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    seen_content: list[str] = []
    deny_result = PolicyResult(
        action=PolicyAction.DENY,
        reason="Input mentions Canada",
    )

    async def _eval(ctx: Any) -> PolicyResult:
        """
        Capture the content evaluated by the policy engine.

        :param ctx: Evaluation context built by the route helper.
        :returns: Deny verdict so the test can verify propagation.
        """
        seen_content.append(ctx.content)
        return deny_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        mock_engine.apply_label_writes = lambda x: None

        result = await _evaluate_input_policy(
            "sess_1",
            conv,
            policy_body,
            conv_store,
            agent_store,
            None,
        )

    assert seen_content == ["/grill-me review Canada rollout"]
    assert result["verdict"] == "deny"
    assert result["reason"] == "Input mentions Canada"


@pytest.mark.asyncio
async def test_input_no_guardrails_skips_policy():
    """When the agent spec has no guardrails, INPUT policy
    is skipped and the message is persisted with allow verdict.
    """
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_user_message_body()

    spec = _make_spec_no_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")

    with (
        patch(_CACHE_PATCH) as mock_cache,
    ):
        mock_cache.return_value.load.return_value = loaded
        result = await _evaluate_input_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result is None


@pytest.mark.asyncio
async def test_input_empty_text_skips_policy():
    """When the user message has no text content, INPUT policy
    is skipped (nothing to evaluate).
    """
    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _FakeBody(
        type="message",
        data={"role": "user", "content": []},
    )

    result = await _evaluate_input_policy(
        "sess_1",
        conv,
        body,
        conv_store,
        agent_store,
        None,
    )

    assert result is None


# ── OUTPUT policy tests (step 5.7) ──────────────────────────


def _make_assistant_message_body(
    text: str = "Here is some information.",
) -> _FakeBody:
    """Build an assistant message event body.

    :param text: Assistant message text.
    :returns: A fake body matching SessionEventInput shape
        for an assistant message.
    """
    return _FakeBody(
        type="message",
        data={
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
            "model": "test-agent",
        },
    )


@pytest.mark.asyncio
async def test_output_allow_verdict():
    """OUTPUT policy evaluation returns allow when the engine
    ALLOWs the assistant response.
    """
    from omnigent.server.routes.sessions import _evaluate_output_policy

    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_assistant_message_body("This is a safe response.")

    spec = _make_spec_with_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    allow_result = PolicyResult(action=PolicyAction.ALLOW)

    async def _eval(_ctx: Any) -> PolicyResult:
        return allow_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        mock_engine.apply_label_writes = lambda x: None

        result = await _evaluate_output_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result is None


@pytest.mark.asyncio
async def test_output_deny_replaces_text():
    """OUTPUT policy DENY replaces the assistant text with the
    deny sentinel in the persisted message.
    """
    from omnigent.server.routes.sessions import _evaluate_output_policy

    conv_store = _FakeConversationStore()
    agent_store = _FakeAgentStore(agent=_make_agent())
    conv = conv_store.get_conversation("sess_1")
    body = _make_assistant_message_body("Here is my secret API key: sk-1234")

    spec = _make_spec_with_guardrails()
    loaded = LoadedAgent(spec=spec, workdir="/tmp/fake")
    deny_result = PolicyResult(
        action=PolicyAction.DENY,
        reason="Response contains a secret",
    )

    async def _eval(_ctx: Any) -> PolicyResult:
        return deny_result

    with (
        patch(_CACHE_PATCH) as mock_cache,
        patch(_ENGINE_PATCH) as mock_build,
    ):
        mock_cache.return_value.load.return_value = loaded
        mock_engine = mock_build.return_value
        mock_engine.evaluate = _eval
        mock_engine.apply_label_writes = lambda x: None

        result = await _evaluate_output_policy(
            "sess_1",
            conv,
            body,
            conv_store,
            agent_store,
            None,
        )

    assert result["verdict"] == "deny"
    assert result["reason"] == "Response contains a secret"
    # Verify the _denied_body has the deny sentinel, not the original text.
    denied_body = result["_denied_body"]
    denied_content = denied_body.data.get("content", [])
    denied_text = denied_content[0]["text"]
    assert "[Denied by policy: Response contains a secret]" in denied_text
    assert "sk-1234" not in denied_text
