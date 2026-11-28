"""The message floor must follow the same tool continuation edges as listing and replay."""

import pytest

from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import SessionCountQuery, SessionOrder, SessionQuery, SessionWindow
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.mark.parametrize("source", ["tool", "cli"])
@pytest.mark.parametrize("view", ["started", "recent", "pinned-backfill", "count"])
def test_message_floor_keeps_a_populated_tip_of_an_empty_root(backend, scope_a, source, view):
    db = backend.open(scope_a)
    seed_session(db, "root", source)
    for parent, child in (("root", "mid"), ("mid", "tip")):
        db.lineage.publish_compression_child(
            parent, CompressionChild(child, source), [NewMessage(role="user", content="summary")],
            require_compression_lease=False)
    assert db.sessions.get("root").message_count == 0
    assert db.sessions.get("mid").message_count == db.sessions.get("tip").message_count == 1
    # The floor is per segment, not the sum of unrelated or individually short segments.
    assert db.sessions.list_sessions(SessionQuery(min_message_count=2)) == ()
    assert db.sessions.count_sessions(SessionCountQuery(min_message_count=2, exclude_children=True)) == 0
    before = db.messages.conversation_revision("tip", include_lineage=True)
    assert before.lineage_ids == ("root", "mid", "tip")
    seed_messages(db, "tip", {"role": "assistant", "content": "continued reply"})
    after = db.messages.conversation_revision("tip", include_lineage=True)
    assert after.lineage_ids == before.lineage_ids
    assert after.active_count == before.active_count + 1
    assert after.max_active_id > before.max_active_id
    assert db.lineage.resolve_resume_target("root") == "tip"
    if view == "count":
        assert db.sessions.count_sessions(SessionCountQuery(min_message_count=2, exclude_children=True)) == 1
        return
    db.sessions.set_pinned("root", True)
    query = SessionQuery(
        min_message_count=2, order=SessionOrder.LAST_ACTIVE if view == "recent" else SessionOrder.STARTED_AT,
        include_pinned=view == "pinned-backfill")
    rows = db.sessions.list_sessions(query, SessionWindow(limit=0 if view == "pinned-backfill" else 20))
    assert [row.session.session_id for row in rows] == ["tip"]
    assert rows[0].lineage_root_id == "root"
    assert rows[0].lineage_ids == ("root", "mid", "tip")


@pytest.mark.parametrize("config", [
    None, {}, {"_compressed_from": "elsewhere"}, {"_compressed_from": 1},
    {"_compressed_from": True}, {"_compressed_from": ["1"]}, {"_compressed_from": {"id": "1"}},
    {"_compressed_from": "1", "_branched_from": "1"},
    {"_compressed_from": "1", "_delegate_from": "1"},
    {"_compressed_from": "1", "_reset_from": "1"},
])
def test_message_floor_does_not_borrow_from_a_tool_fork(backend, scope_a, config):
    db = backend.open(scope_a)
    # A numeric-looking id must not let SQLite coerce a non-string marker into a match.
    seed_session(db, "1", "tool", end_reason="compression")
    seed_session(db, "child", "tool", parent_session_id="1", model_config=config)
    seed_messages(db, "child", {"role": "user", "content": "separate conversation"})
    rows = db.sessions.list_sessions(SessionQuery(
        min_message_count=1, include_children=True, project_compression_tips=False))
    assert [row.session.session_id for row in rows] == ["child"]
    assert db.sessions.count_sessions(SessionCountQuery(min_message_count=1)) == 1
    assert db.lineage.lineage("child") == ("child",)
    assert db.messages.conversation_revision("child", include_lineage=True).lineage_ids == ("child",)
