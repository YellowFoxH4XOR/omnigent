"""Tests for :class:`SqlAlchemyPolicyStore`.

Exercises the ``create``, ``get``, ``list_for_session``, ``update``,
and ``delete`` methods against a real SQLite database.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from omnigent.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)
from omnigent.stores.policy_store.sqlalchemy_store import (
    SqlAlchemyPolicyStore,
)


@pytest.fixture()
def store(db_uri: str) -> SqlAlchemyPolicyStore:
    """A fresh :class:`SqlAlchemyPolicyStore` backed by the test SQLite DB.

    :param db_uri: Per-test SQLite URI from the root conftest fixture.
    :returns: A ready-to-use :class:`SqlAlchemyPolicyStore` instance.
    """
    return SqlAlchemyPolicyStore(db_uri)


@pytest.fixture()
def session_id(db_uri: str) -> str:
    """Create a real conversation row and return its ID.

    Required because ``policies.session_id`` has a FK to
    ``conversations.id`` — raw strings fail the FK check.

    :param db_uri: Per-test SQLite URI.
    :returns: A conversation ID, e.g. ``"conv_abc123"``.
    """
    conv_store = SqlAlchemyConversationStore(db_uri)
    return conv_store.create_conversation().id


@pytest.fixture()
def other_session_id(db_uri: str) -> str:
    """Create a second conversation row for cross-session isolation tests.

    :param db_uri: Per-test SQLite URI.
    :returns: A conversation ID different from :func:`session_id`.
    """
    conv_store = SqlAlchemyConversationStore(db_uri)
    return conv_store.create_conversation().id


# ── create_session_policy ───────────────────────────────────────────────────


def test_create_returns_policy_with_correct_fields(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``create_session_policy`` returns a Policy with all fields echoed back.

    Verifies that the entity round-trips through the ORM layer without
    loss — session_id, handler, and nullable prompt-policy fields all
    map correctly.
    """
    policy = store.create(
        policy_id="pol_test1",
        session_id=session_id,
        name="block_push",
        type="python",
        handler="github_mcp_policy.block_push",
    )

    assert policy.id == "pol_test1"
    assert policy.session_id == session_id
    assert policy.name == "block_push"
    assert policy.type == "python"
    assert policy.handler == "github_mcp_policy.block_push"
    assert policy.enabled is True
    assert policy.created_at > 0
    assert policy.updated_at is None


def test_create_url_type(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``create_session_policy`` with ``type="url"`` stores an HTTP endpoint handler."""
    policy = store.create(
        policy_id="pol_url1",
        session_id=session_id,
        name="external_eval",
        type="url",
        handler="https://example.com/policies/eval",
    )

    assert policy.type == "url"
    assert policy.handler == "https://example.com/policies/eval"


def test_create_duplicate_name_raises(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``create_session_policy`` with a duplicate ``(session_id, name)`` raises IntegrityError."""
    store.create(
        policy_id="pol_dup1",
        session_id=session_id,
        name="dup_policy",
        type="python",
        handler="mod.func",
    )
    with pytest.raises(IntegrityError):
        store.create(
            policy_id="pol_dup2",
            session_id=session_id,
            name="dup_policy",
            type="python",
            handler="mod.func2",
        )


def test_create_same_name_different_sessions(
    store: SqlAlchemyPolicyStore,
    session_id: str,
    other_session_id: str,
) -> None:
    """Two sessions may have policies with the same name."""
    p1 = store.create(
        policy_id="pol_s1",
        session_id=session_id,
        name="shared_name",
        type="python",
        handler="mod.func",
    )
    p2 = store.create(
        policy_id="pol_s2",
        session_id=other_session_id,
        name="shared_name",
        type="python",
        handler="mod.func",
    )

    assert p1.id != p2.id
    assert p1.name == p2.name == "shared_name"


# ── get_session_policy ──────────────────────────────────────────────────────


def test_get_returns_policy(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``get_session_policy`` returns the policy when it belongs to the session."""
    created = store.create(
        policy_id="pol_get1",
        session_id=session_id,
        name="get_policy",
        type="python",
        handler="mod.func",
    )
    fetched = store.get("pol_get1", session_id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.name == "get_policy"


def test_get_returns_none_for_missing(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``get_session_policy`` returns ``None`` when the policy does not exist."""
    assert store.get("pol_nonexistent", session_id) is None


def test_get_returns_none_for_wrong_session(
    store: SqlAlchemyPolicyStore,
    session_id: str,
    other_session_id: str,
) -> None:
    """``get_session_policy`` returns ``None`` for a different session.

    Prevents cross-session data leakage.
    """
    store.create(
        policy_id="pol_wrong",
        session_id=session_id,
        name="owned_policy",
        type="python",
        handler="mod.func",
    )
    assert store.get("pol_wrong", other_session_id) is None


# ── list_for_session ────────────────────────────────────────────────────────


def test_list_for_session_returns_policies_in_order(
    store: SqlAlchemyPolicyStore,
    session_id: str,
    other_session_id: str,
) -> None:
    """``list_for_session`` returns policies ordered by ``created_at ASC``.

    Also verifies session isolation — policies from other sessions
    must not appear.
    """
    store.create(
        policy_id="pol_list1",
        session_id=session_id,
        name="first",
        type="python",
        handler="mod.a",
    )
    store.create(
        policy_id="pol_list2",
        session_id=session_id,
        name="second",
        type="url",
        handler="https://example.com",
    )
    # Different session — should not appear.
    store.create(
        policy_id="pol_other",
        session_id=other_session_id,
        name="other",
        type="python",
        handler="mod.b",
    )

    policies = store.list_for_session(session_id)

    assert len(policies) == 2
    assert policies[0].name == "first"
    assert policies[1].name == "second"


def test_list_for_session_empty(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``list_for_session`` returns an empty list for a session with no policies."""
    assert store.list_for_session(session_id) == []


# ── update_session_policy ───────────────────────────────────────────────────


def test_update_changes_name(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``update_session_policy`` with ``name=`` changes the name and bumps ``updated_at``."""
    store.create(
        policy_id="pol_upd1",
        session_id=session_id,
        name="old_name",
        type="python",
        handler="mod.func",
    )
    updated = store.update("pol_upd1", session_id, name="new_name")

    assert updated is not None
    assert updated.name == "new_name"
    assert updated.updated_at is not None
    assert updated.updated_at > 0


def test_update_changes_enabled(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``update_session_policy`` with ``enabled=False`` disables the policy."""
    store.create(
        policy_id="pol_upd2",
        session_id=session_id,
        name="toggle_policy",
        type="python",
        handler="mod.func",
    )
    updated = store.update("pol_upd2", session_id, enabled=False)

    assert updated is not None
    assert updated.enabled is False


def test_update_changes_handler(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``update_session_policy`` with ``handler=`` changes the handler path."""
    store.create(
        policy_id="pol_upd3",
        session_id=session_id,
        name="handler_policy",
        type="python",
        handler="mod.old_func",
    )
    updated = store.update("pol_upd3", session_id, handler="mod.new_func")

    assert updated is not None
    assert updated.handler == "mod.new_func"


def test_update_noop_does_not_bump_timestamp(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``update_session_policy`` with no changes does not bump ``updated_at``."""
    store.create(
        policy_id="pol_noop",
        session_id=session_id,
        name="noop_policy",
        type="python",
        handler="mod.func",
    )
    updated = store.update("pol_noop", session_id)

    assert updated is not None
    assert updated.updated_at is None


def test_update_returns_none_for_missing(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``update_session_policy`` returns ``None`` when the policy does not exist."""
    assert store.update("pol_missing", session_id, name="x") is None


def test_update_returns_none_for_wrong_session(
    store: SqlAlchemyPolicyStore,
    session_id: str,
    other_session_id: str,
) -> None:
    """``update_session_policy`` returns ``None`` for a different session."""
    store.create(
        policy_id="pol_xsess",
        session_id=session_id,
        name="xsess_policy",
        type="python",
        handler="mod.func",
    )
    assert store.update("pol_xsess", other_session_id, enabled=False) is None


# ── delete_session_policy ───────────────────────────────────────────────────


def test_delete_removes_policy(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``delete_session_policy`` removes the policy and returns ``True``."""
    store.create(
        policy_id="pol_del1",
        session_id=session_id,
        name="to_delete",
        type="python",
        handler="mod.func",
    )
    assert store.delete("pol_del1", session_id) is True
    assert store.get("pol_del1", session_id) is None


def test_delete_idempotent(
    store: SqlAlchemyPolicyStore,
    session_id: str,
) -> None:
    """``delete_session_policy`` on a missing policy returns ``False``."""
    assert store.delete("pol_missing", session_id) is False


def test_delete_wrong_session(
    store: SqlAlchemyPolicyStore,
    session_id: str,
    other_session_id: str,
) -> None:
    """``delete_session_policy`` returns ``False`` for a different session."""
    store.create(
        policy_id="pol_del_x",
        session_id=session_id,
        name="xdel_policy",
        type="python",
        handler="mod.func",
    )
    assert store.delete("pol_del_x", other_session_id) is False
    assert store.get("pol_del_x", session_id) is not None
