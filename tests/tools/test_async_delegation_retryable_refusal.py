"""Retryable refusals must not consume the durable delivery budget, and a
completion dropped by budget exhaustion must be rehabilitated once on the next
restart while its result is still fresh.

Incident (2026-09-06): a WebUI in its ``agent_runtime_stale`` barrier answered
409 ``retryable=True`` to eight wake-up attempts in 36 s; the claim/release
cycle counted each as a real failure and the review verdict was terminally
``dropped`` while the origin session still existed.
"""
from __future__ import annotations

import json
import queue
import time

import pytest


def _persist(ad, delegation_id: str, *, session_key: str = "S1") -> None:
    ad._persist_dispatch({
        "delegation_id": delegation_id, "goal": "g", "context": None, "toolsets": None,
        "role": "leaf", "model": "m", "session_key": session_key, "origin_ui_session_id": session_key,
        "parent_session_id": session_key, "status": "running", "dispatched_at": time.time() - 5.0,
        "completed_at": None, "interrupt_fn": None,
    })
    evt = {
        "type": "async_delegation", "delegation_id": delegation_id, "session_key": session_key,
        "origin_ui_session_id": session_key, "parent_session_id": session_key, "goal": "g",
        "status": "completed", "summary": "verdict", "dispatched_at": time.time() - 5.0,
        "completed_at": time.time(),
    }
    ad._persist_completion(evt, {"status": "completed", "summary": "verdict"})


def _row(ad, delegation_id: str) -> tuple:
    with ad._connect() as conn:
        return conn.execute(
            "SELECT delivery_state, delivery_attempts, drop_reason FROM async_delegations WHERE delegation_id=?",
            (delegation_id,),
        ).fetchone()


@pytest.fixture
def ad(tmp_path, monkeypatch):
    import tools.async_delegation as module

    monkeypatch.setattr(module, "_db_path", lambda: tmp_path / "async_delegations.db")
    return module


def test_retryable_release_refunds_the_claimed_attempt(ad):
    _persist(ad, "d-retry")
    for _ in range(ad._MAX_DELIVERY_ATTEMPTS * 3):
        claim = ad.claim_event_delivery({"type": "async_delegation", "delegation_id": "d-retry"}, "webui")
        assert claim, "a retryable refusal must never exhaust the budget"
        assert ad.release_completion_delivery("d-retry", claim, retryable=True) is True
    state, attempts, reason = _row(ad, "d-retry")
    assert state == "pending"
    assert attempts == 0
    assert reason is None


def test_non_retryable_release_still_converges_to_dropped(ad):
    _persist(ad, "d-hard")
    for _ in range(ad._MAX_DELIVERY_ATTEMPTS):
        claim = ad.claim_event_delivery({"type": "async_delegation", "delegation_id": "d-hard"}, "webui")
        assert claim
        ad.release_completion_delivery("d-hard", claim)
    state, attempts, reason = _row(ad, "d-hard")
    assert state == "dropped"
    assert attempts == ad._MAX_DELIVERY_ATTEMPTS
    assert reason == "attempts_exhausted"


def test_retryable_release_is_bounded_by_replay_age_not_by_count(ad):
    _persist(ad, "d-stale")
    with ad._connect() as conn:
        conn.execute("UPDATE async_delegations SET completed_at=? WHERE delegation_id='d-stale'",
                     (time.time() - ad._MAX_COMPLETION_REPLAY_AGE_S - 60,))
    claim = ad.claim_event_delivery({"type": "async_delegation", "delegation_id": "d-stale"}, "webui")
    assert claim
    ad.release_completion_delivery("d-stale", claim, retryable=True)
    state, _attempts, reason = _row(ad, "d-stale")
    assert state == "dropped"
    assert reason == "replay_age"


def test_restart_rehabilitates_a_fresh_budget_exhausted_drop_once(ad):
    _persist(ad, "d-lost")
    for _ in range(ad._MAX_DELIVERY_ATTEMPTS):
        claim = ad.claim_event_delivery({"type": "async_delegation", "delegation_id": "d-lost"}, "webui")
        ad.release_completion_delivery("d-lost", claim)
    assert _row(ad, "d-lost")[0] == "dropped"

    q = queue.Queue()
    assert ad.restore_undelivered_completions(q) == 1
    evt = q.get_nowait()
    assert evt["delegation_id"] == "d-lost" and evt["restored"] is True
    state, attempts, reason = _row(ad, "d-lost")
    assert (state, attempts, reason) == ("pending", 0, None)

    # A second sweep in the same process state finds nothing new to rehabilitate:
    # the row is pending (replayed as usual), not re-dropped and re-rehabilitated.
    q2 = queue.Queue()
    assert ad.restore_undelivered_completions(q2) == 1
    assert q2.get_nowait()["delegation_id"] == "d-lost"


def test_restart_never_rehabilitates_target_gone_or_aged_drops(ad):
    _persist(ad, "d-gone")
    claim = ad.claim_event_delivery({"type": "async_delegation", "delegation_id": "d-gone"}, "webui")
    assert ad.drop_completion_delivery("d-gone", claim) is True
    assert _row(ad, "d-gone")[2] == "target_gone"

    _persist(ad, "d-old")
    for _ in range(ad._MAX_DELIVERY_ATTEMPTS):
        c = ad.claim_event_delivery({"type": "async_delegation", "delegation_id": "d-old"}, "webui")
        ad.release_completion_delivery("d-old", c)
    with ad._connect() as conn:
        conn.execute("UPDATE async_delegations SET completed_at=? WHERE delegation_id='d-old'",
                     (time.time() - ad._MAX_COMPLETION_REPLAY_AGE_S - 60,))

    q = queue.Queue()
    assert ad.restore_undelivered_completions(q) == 0
    assert _row(ad, "d-gone")[0] == "dropped"
    assert _row(ad, "d-old")[0] == "dropped"


def test_legacy_rows_without_drop_reason_are_rehabilitated_by_exhausted_signature(ad):
    """Rows dropped before this column existed carry NULL: only the exhausted
    signature (attempts >= budget) is trusted, a low-attempt drop is left alone."""
    _persist(ad, "d-legacy-exhausted")
    _persist(ad, "d-legacy-gone")
    with ad._connect() as conn:
        conn.execute("UPDATE async_delegations SET delivery_state='dropped', delivery_attempts=?, drop_reason=NULL "
                     "WHERE delegation_id='d-legacy-exhausted'", (ad._MAX_DELIVERY_ATTEMPTS,))
        conn.execute("UPDATE async_delegations SET delivery_state='dropped', delivery_attempts=1, drop_reason=NULL "
                     "WHERE delegation_id='d-legacy-gone'")
    q = queue.Queue()
    assert ad.restore_undelivered_completions(q) == 1
    assert q.get_nowait()["delegation_id"] == "d-legacy-exhausted"
    assert _row(ad, "d-legacy-gone")[0] == "dropped"


def test_release_event_delivery_forwards_retryable_flag(ad):
    _persist(ad, "d-evt")
    evt = {"type": "async_delegation", "delegation_id": "d-evt"}
    claim = ad.claim_event_delivery(evt, "webui")
    ad.release_event_delivery(evt, claim, retryable=True)
    assert _row(ad, "d-evt")[1] == 0
    payload = json.loads(json.dumps(evt))
    assert payload["delegation_id"] == "d-evt"
