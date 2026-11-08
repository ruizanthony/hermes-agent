"""Cold restore of an idle-retired local conversation whose route still names an older lineage segment.

A local conversation compressed while the loaded SessionStore missed the split (the compression was
published by another process, e.g. a TUI holding the conversation) keeps a route entry naming the
compressed parent, while the creation receipt names the new tip. Once the owner retires the
conversation at idle, every ``session.resume`` takes the cold branch of ``restore_local_session``:
``_stray_route_target`` excuses only ids *outside* the lineage, so the older segment of the very same
conversation answered ``admission_conflict`` forever and the conversation could no longer be opened.
The live branch already accepts a lineage member and rebinds it to the receipt's target; the cold
branch now does the same. Real authority, real SQLite, real compression publish.
"""
from types import SimpleNamespace

import pytest


def _runner(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return runner


def _compress(db, parent, child, source):
    history = [{'role': 'user', 'content': 'retained'}, {'role': 'assistant', 'content': 'history'}]
    assert db.try_acquire_compression_lock(parent, 'fixture')
    try:
        db.publish_compression_child(parent_session_id=parent, child_session_id=child,
                                     source=source, messages=history, compression_lock_holder='fixture')
    finally:
        db.release_compression_lock(parent, 'fixture')


async def _resume(authority, user_id, sid):
    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': user_id, 'profile_id': 'fixture', 'instance_id': authority.instance_id,
        'capabilities': ['session:create', 'session:read', 'session:control']})
    return await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})


async def _retired_with_stale_route(tmp_path, monkeypatch, segment, *, yolo=None):
    """owner -> child-1 -> child-2, retired at idle, the loaded route still naming *segment*."""
    from gateway import run
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from gateway.session_retirement import retire_idle
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    authority = await initialize_session_authority(_runner(tmp_path), profile_id='fixture', instance_id='first')
    actor = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(authority, actor, dict(request_id='stale-route', source='gui', cwd=str(tmp_path),
                                                      model='frozen', toolsets=[]))
    live = authority.sessions[ref.session_id]
    store = authority.runner.session_store
    for parent, child in ((ref.session_id, 'child-1'), ('child-1', 'child-2')):
        store._record_gateway_session_peer(parent, live.route, live.source)
        _compress(authority.db, parent, child, authority.db.get_session(parent)['source'])
        store._record_gateway_session_peer(child, live.route, live.source)
    assert retire_idle(authority, ref.session_id, live, 'child-2')
    if segment == 'owner':
        segment = ref.session_id
    elif segment == 'foreign':
        authority.db.create_session('foreign', source='local')
        authority.db.append_message('foreign', 'user', 'not this conversation')
    with store._lock:
        entry = store._entry_locked(live.route)
        entry.session_id = segment  # the split this loaded store never saw
        if yolo is not None:
            entry.yolo = yolo
    return SimpleNamespace(authority=authority, owner=ref.session_id, route=live.route, store=store)


@pytest.mark.asyncio
@pytest.mark.parametrize('segment', ['owner', 'child-1'])
async def test_cold_restore_rebinds_an_older_lineage_segment_to_the_receipt_tip(tmp_path, monkeypatch, segment):
    retired = await _retired_with_stale_route(tmp_path, monkeypatch, segment)
    reply = await _resume(retired.authority, 'owner', retired.owner)
    assert 'result' in reply, reply
    assert reply['result']['session_id'] == retired.owner
    assert [m['content'] for m in reply['result']['messages']] == ['retained', 'history']
    assert retired.store._entries[retired.route].session_id == 'child-2'
    assert 'result' in await _resume(retired.authority, 'owner', retired.owner)


@pytest.mark.asyncio
async def test_cold_restore_keeps_the_route_yolo_toggle_of_the_rebound_entry(tmp_path, monkeypatch):
    retired = await _retired_with_stale_route(tmp_path, monkeypatch, 'owner', yolo=False)
    assert 'result' in await _resume(retired.authority, 'owner', retired.owner)
    assert retired.store._entries[retired.route].yolo is False


@pytest.mark.asyncio
async def test_a_foreign_route_target_with_history_still_refuses(tmp_path, monkeypatch):
    retired = await _retired_with_stale_route(tmp_path, monkeypatch, 'foreign')
    reply = await _resume(retired.authority, 'owner', retired.owner)
    assert reply['error']['message'] == 'admission_conflict', reply


@pytest.mark.asyncio
async def test_another_principal_still_refused(tmp_path, monkeypatch):
    retired = await _retired_with_stale_route(tmp_path, monkeypatch, 'owner')
    reply = await _resume(retired.authority, 'intruder', retired.owner)
    assert reply['error']['message'] == 'permission_denied', reply
