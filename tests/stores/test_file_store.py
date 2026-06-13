"""Tests for SqlAlchemyFileStore."""

from __future__ import annotations

import pytest

from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore


@pytest.fixture()
def file_store(db_uri: str) -> SqlAlchemyFileStore:
    return SqlAlchemyFileStore(db_uri)


def test_create_and_get(file_store: SqlAlchemyFileStore) -> None:
    f = file_store.create(filename="data.csv", bytes=1024)
    assert f.id.startswith("file_")
    assert f.filename == "data.csv"
    assert f.bytes == 1024

    fetched = file_store.get(f.id)
    assert fetched is not None
    assert fetched.filename == "data.csv"


def test_get_nonexistent(file_store: SqlAlchemyFileStore) -> None:
    assert file_store.get("file_nonexistent") is None


def test_create_with_content_type(file_store: SqlAlchemyFileStore) -> None:
    f = file_store.create(
        filename="img.png",
        bytes=2048,
        content_type="image/png",
    )
    assert f.content_type == "image/png"


def test_delete(file_store: SqlAlchemyFileStore) -> None:
    f = file_store.create(filename="temp.txt", bytes=10)
    assert file_store.delete(f.id) is True
    assert file_store.get(f.id) is None
    assert file_store.delete(f.id) is False


def test_list_pagination(file_store: SqlAlchemyFileStore) -> None:
    for i in range(4):
        file_store.create(filename=f"f{i}.txt", bytes=i)

    page1 = file_store.list(limit=2)
    assert len(page1.data) == 2
    assert page1.has_more is True

    page2 = file_store.list(limit=2, after=page1.last_id)
    assert len(page2.data) == 2
    assert page2.has_more is False


def test_list_order_asc(file_store: SqlAlchemyFileStore) -> None:
    for i in range(3):
        file_store.create(filename=f"f{i}.txt", bytes=i)
    page_desc = file_store.list(order="desc")
    page_asc = file_store.list(order="asc")
    assert [f.id for f in page_asc.data] == list(reversed([f.id for f in page_desc.data]))


def test_list_asc_with_after_cursor(file_store: SqlAlchemyFileStore) -> None:
    for i in range(5):
        file_store.create(filename=f"f{i}.txt", bytes=i)

    page1 = file_store.list(limit=2, order="asc")
    page2 = file_store.list(limit=2, order="asc", after=page1.last_id)
    page3 = file_store.list(limit=2, order="asc", after=page2.last_id)

    all_ids = [f.id for f in page1.data + page2.data + page3.data]
    full_asc = file_store.list(limit=100, order="asc")
    assert all_ids == [f.id for f in full_asc.data]


# ── Phase 1c: session-scoped file store methods ─────────────────


def test_create_for_session(file_store: SqlAlchemyFileStore) -> None:
    """Session-scoped create records session_id on the file."""
    f = file_store.create(
        session_id="conv_abc",
        filename="report.pdf",
        bytes=5000,
        content_type="application/pdf",
    )
    assert f.id.startswith("file_")
    assert f.session_id == "conv_abc"
    assert f.filename == "report.pdf"
    assert f.bytes == 5000


def test_get_for_session_validates_ownership(
    file_store: SqlAlchemyFileStore,
) -> None:
    """get_for_session returns None if file belongs to another session."""
    f = file_store.create(
        session_id="conv_abc",
        filename="owned.txt",
        bytes=10,
    )
    assert file_store.get(f.id, session_id="conv_abc") is not None
    assert file_store.get(f.id, session_id="conv_other") is None


def test_list_for_session_scopes_to_session(
    file_store: SqlAlchemyFileStore,
) -> None:
    """list_for_session only returns files owned by that session."""
    file_store.create("a1.txt", 1, session_id="conv_a")
    file_store.create("a2.txt", 2, session_id="conv_a")
    file_store.create("b1.txt", 3, session_id="conv_b")
    file_store.create("global.txt", 4)

    page_a = file_store.list(session_id="conv_a")
    assert len(page_a.data) == 2
    assert all(f.session_id == "conv_a" for f in page_a.data)

    page_b = file_store.list(session_id="conv_b")
    assert len(page_b.data) == 1
    assert page_b.data[0].filename == "b1.txt"


def test_delete_for_session_validates_ownership(
    file_store: SqlAlchemyFileStore,
) -> None:
    """delete_for_session refuses to delete a file from another session."""
    f = file_store.create("mine.txt", 10, session_id="conv_abc")
    assert file_store.delete(f.id, session_id="conv_other") is False
    assert file_store.get(f.id) is not None
    assert file_store.delete(f.id, session_id="conv_abc") is True
    assert file_store.get(f.id) is None


def test_delete_all_for_session(
    file_store: SqlAlchemyFileStore,
) -> None:
    """delete_all_for_session removes all session files and returns ids."""
    f1 = file_store.create("a.txt", 1, session_id="conv_abc")
    f2 = file_store.create("b.txt", 2, session_id="conv_abc")
    file_store.create("c.txt", 3, session_id="conv_other")
    global_f = file_store.create("global.txt", 4)

    deleted_ids = file_store.delete_all_for_session("conv_abc")
    assert set(deleted_ids) == {f1.id, f2.id}
    assert file_store.get(f1.id) is None
    assert file_store.get(f2.id) is None
    other_page = file_store.list(session_id="conv_other")
    assert file_store.get(other_page.data[0].id, session_id="conv_other") is not None
    assert file_store.get(global_f.id) is not None
