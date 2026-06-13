"""
Integration test: ``LEVEL_READ`` callers get policy verdicts without session mutation.

Uses a real ``SqlAlchemyPermissionStore`` so ``UnifiedAuthProvider``
resolves ``X-Forwarded-Email`` headers to permission levels. An owner
creates a session with a label-writing policy, then a ``LEVEL_READ``
collaborator evaluates the same session. The collaborator receives the
verdict (ALLOW with ``set_labels`` populated), but the session's
persisted labels are unchanged afterward. A second test confirms that
an ``LEVEL_EDIT`` caller still persists as before.

Uses the shared ``mock_llm`` / ``runtime_init`` / ``db_uri`` fixtures
from ``tests/server/conftest.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from omnigent.runtime import get_caps
from omnigent.runtime.agent_cache import AgentCache
from omnigent.runtime.caps import RuntimeCaps
from omnigent.server.app import create_app
from omnigent.server.auth import LEVEL_EDIT, LEVEL_READ
from omnigent.spec.types import FunctionPolicySpec, FunctionRef
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.artifact_store.local import LocalArtifactStore
from omnigent.stores.comment_store.sqlalchemy_store import SqlAlchemyCommentStore
from omnigent.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore
from omnigent.stores.permission_store.sqlalchemy_store import SqlAlchemyPermissionStore
from tests.server.conftest import ControllableMockClient
from tests.server.helpers import create_test_agent

pytestmark = pytest.mark.asyncio


# ── Policy callable ──────────────────────────────────────────────


def _allow_and_label(event: dict[str, Any]) -> dict[str, Any]:
    """
    Policy that ALLOWs every event and writes a label.

    Always returns ``set_labels: {"evaluated": "true"}`` so we can
    verify whether the label was actually persisted.

    :param event: V0 event dict.
    :returns: ALLOW with ``set_labels``.
    """
    return {
        "result": "ALLOW",
        "set_labels": {"evaluated": "true"},
    }


# ── Helpers ──────────────────────────────────────────────────────

OWNER = "owner@example.com"
READER = "reader@example.com"
EDITOR = "editor@example.com"


def _tool_call_request(tool_name: str = "Bash") -> dict[str, Any]:
    """
    Build a PHASE_TOOL_CALL EvaluationRequest.

    :param tool_name: Tool name, e.g. ``"Bash"``.
    :returns: EvaluationRequest JSON dict.
    """
    return {
        "event": {
            "type": "PHASE_TOOL_CALL",
            "target": "",
            "data": {"name": tool_name, "arguments": {}},
            "context": {},
        },
    }


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def auth_app(
    runtime_init: None,
    db_uri: str,
    tmp_path: Path,
) -> FastAPI:
    """
    App with ``permission_store`` enabled so auth is active.

    :param runtime_init: Fixture that initializes the runtime with a mock LLM.
    :param db_uri: Per-test SQLite URI.
    :param tmp_path: Pytest temp dir for artifacts.
    :returns: A :class:`FastAPI` instance with auth and policy routes active.
    """
    from omnigent.server.auth import UnifiedAuthProvider

    artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
    return create_app(
        agent_store=SqlAlchemyAgentStore(db_uri),
        file_store=SqlAlchemyFileStore(db_uri),
        conversation_store=SqlAlchemyConversationStore(db_uri),
        artifact_store=artifact_store,
        agent_cache=AgentCache(
            artifact_store=artifact_store,
            cache_dir=tmp_path / "cache",
        ),
        comment_store=SqlAlchemyCommentStore(db_uri),
        permission_store=SqlAlchemyPermissionStore(db_uri),
        auth_provider=UnifiedAuthProvider(source="header"),
    )


@pytest_asyncio.fixture()
async def auth_client(
    auth_app: FastAPI,
    mock_llm: ControllableMockClient,
    tmp_path: Path,
) -> AsyncIterator[httpx.AsyncClient]:
    """
    Async HTTP client wired to the auth-enabled app.

    :param auth_app: FastAPI app with permission store.
    :param mock_llm: Controllable mock LLM -- released on teardown.
    :param tmp_path: Pytest temp dir for the harness process manager.
    :yields: A ready-to-use :class:`httpx.AsyncClient`.
    """
    from omnigent.runtime import set_harness_process_manager
    from omnigent.runtime.harnesses.process_manager import HarnessProcessManager

    pm = HarnessProcessManager(tmp_parent=tmp_path / "harness_pm")
    await pm.start()
    set_harness_process_manager(pm)

    transport = httpx.ASGITransport(app=auth_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    mock_llm.release_all()
    set_harness_process_manager(None)
    await pm.shutdown()


async def _create_session_as(
    client: httpx.AsyncClient,
    user: str,
    agent_id: str,
) -> str:
    """
    Create a session via the API as the given user.

    :param client: Test HTTP client.
    :param user: User email for ``X-Forwarded-Email``.
    :param agent_id: Agent to bind.
    :returns: New session id.
    """
    resp = await client.post(
        "/v1/sessions",
        json={"agent_id": agent_id},
        headers={"X-Forwarded-Email": user},
    )
    assert resp.status_code == 201, f"create session failed: {resp.status_code} {resp.text}"
    return resp.json()["id"]


def _grant_access(db_uri: str, user: str, session_id: str, level: int) -> None:
    """
    Grant a permission level to a user on a session.

    :param db_uri: SQLite connection URI.
    :param user: User email.
    :param session_id: Session (conversation) id.
    :param level: Permission level, e.g. ``LEVEL_READ``.
    """
    perm_store = SqlAlchemyPermissionStore(db_uri)
    perm_store.ensure_user(user)
    perm_store.grant(user, session_id, level)


def _get_labels(db_uri: str, session_id: str) -> dict[str, str]:
    """
    Read persisted labels from the conversation store.

    :param db_uri: SQLite connection URI.
    :param session_id: Session (conversation) id.
    :returns: The conversation's label dict.
    """
    conv_store = SqlAlchemyConversationStore(db_uri)
    conv = conv_store.get_conversation(session_id)
    assert conv is not None, f"session {session_id!r} not found"
    return dict(conv.labels)


# ── Tests ────────────────────────────────────────────────────────


async def test_read_only_caller_gets_verdict_but_no_label_mutation(
    auth_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    db_uri: str,
) -> None:
    """
    A LEVEL_READ collaborator receives the policy verdict but session
    labels are NOT persisted.

    The label-writing policy returns ``set_labels: {"evaluated": "true"}``.
    The LEVEL_READ caller sees ALLOW in the response, but the session's
    persisted ``conversation_labels`` remains empty afterward. This
    verifies the ``read_only`` guard in the route and the engine.
    """
    labeling_policy = FunctionPolicySpec(
        name="admin__labeler",
        on=None,
        function=FunctionRef(path=f"{__name__}._allow_and_label"),
    )
    original_caps = get_caps()
    patched_caps = RuntimeCaps(
        execution_timeout=original_caps.execution_timeout,
        default_policies=[labeling_policy],
    )
    monkeypatch.setattr(
        "omnigent.server.routes.sessions.get_caps",
        lambda: patched_caps,
    )

    # Owner creates the agent and session.
    agent = await create_test_agent(auth_client, user=OWNER)
    session_id = await _create_session_as(auth_client, OWNER, agent["id"])

    # Grant LEVEL_READ to reader.
    _grant_access(db_uri, READER, session_id, LEVEL_READ)

    # Snapshot labels before.
    labels_before = _get_labels(db_uri, session_id)
    assert labels_before.get("evaluated") is None, "label should not exist before evaluation"

    # Reader evaluates — should get ALLOW but no persistence.
    resp = await auth_client.post(
        f"/v1/sessions/{session_id}/policies/evaluate",
        json=_tool_call_request("Read"),
        headers={"X-Forwarded-Email": READER},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"] == "POLICY_ACTION_ALLOW"

    # Labels must NOT have been persisted.
    labels_after = _get_labels(db_uri, session_id)
    assert labels_after.get("evaluated") is None, (
        "LEVEL_READ caller must not mutate session labels via policy evaluate"
    )


async def test_edit_caller_persists_labels_as_before(
    auth_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    db_uri: str,
) -> None:
    """
    A LEVEL_EDIT (or higher) caller's policy evaluation still persists labels.

    Ensures the ``read_only`` guard only fires for sub-edit levels and
    does not regress the normal write path.
    """
    labeling_policy = FunctionPolicySpec(
        name="admin__labeler",
        on=None,
        function=FunctionRef(path=f"{__name__}._allow_and_label"),
    )
    original_caps = get_caps()
    patched_caps = RuntimeCaps(
        execution_timeout=original_caps.execution_timeout,
        default_policies=[labeling_policy],
    )
    monkeypatch.setattr(
        "omnigent.server.routes.sessions.get_caps",
        lambda: patched_caps,
    )

    # Owner creates the agent and session.
    agent = await create_test_agent(auth_client, user=OWNER)
    session_id = await _create_session_as(auth_client, OWNER, agent["id"])

    # Grant LEVEL_EDIT to editor.
    _grant_access(db_uri, EDITOR, session_id, LEVEL_EDIT)

    # Editor evaluates — should persist labels.
    resp = await auth_client.post(
        f"/v1/sessions/{session_id}/policies/evaluate",
        json=_tool_call_request("Read"),
        headers={"X-Forwarded-Email": EDITOR},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["result"] == "POLICY_ACTION_ALLOW"

    # Labels SHOULD have been persisted.
    labels_after = _get_labels(db_uri, session_id)
    assert labels_after.get("evaluated") == "true", (
        "LEVEL_EDIT caller's policy evaluation must still persist labels"
    )
