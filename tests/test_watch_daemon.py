"""Integration tests for SQLite-to-scheduler watch reconciliation."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest

from insto._redact import register_secret
from insto.models import WatchSpec
from insto.service.history import HistoryStore
from insto.service.watch import TickFn, WatchManager
from insto.service.watch_daemon import (
    WatchDaemon,
    WatchExecutorRole,
    estimate_watch_load,
    initial_watch_delay,
    startup_offsets,
    wants_first_check_now,
)
from insto.service.watch_lock import WatchProcessLock


@pytest.fixture
def history(tmp_path: Path) -> Generator[HistoryStore, None, None]:
    store = HistoryStore(tmp_path / "store.db")
    try:
        yield store
    finally:
        store.close()


def _manager(history: HistoryStore, *, repl: bool = False) -> WatchManager:
    return WatchManager(
        WatchProcessLock(history.path),
        release_when_empty=repl,
        now=lambda: 1_000,
    )


def _ticks(calls: list[str]) -> Callable[[str], TickFn]:
    def factory(user: str) -> TickFn:
        async def tick() -> None:
            calls.append(user)

        return tick

    return factory


def _already_checked(store: HistoryStore, user: str, interval: int = 300) -> WatchSpec:
    """Register a watch that the daemon must not check immediately.

    The headless daemon gives a clean never-checked registration a zero first
    delay, so a test about anything else records one success first and keeps a
    full interval before the scheduled loop would tick.
    """
    spec = store.register_watch(user, interval).spec
    assert spec is not None
    assert store.update_watch_state(spec, last_ok=int(time.time()))
    checked = store.get_watch(user)
    assert checked is not None
    return checked


def _persist_state(
    store: HistoryStore,
    spec: WatchSpec,
    *,
    last_ok: int | None,
) -> WatchSpec:
    assert store.update_watch_state(spec, last_ok=last_ok)
    current = store.get_watch(spec.user)
    assert current is not None
    return current


def _seed_snapshot(store: HistoryStore, *, target: str, captured_at: int) -> None:
    with store._lock:
        store._conn.execute(
            """
            INSERT INTO snapshots(
                target_pk, captured_at, profile_fields_json,
                last_post_pks_json, avatar_url_hash, banner_url_hash
            ) VALUES (?, ?, ?, ?, NULL, NULL)
            """,
            (target, captured_at, json.dumps({"username": target}), json.dumps([])),
        )


def _snapshot_count(store: HistoryStore, target: str) -> int:
    with store._lock:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE target_pk = ?", (target,)
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_initial_delay_and_startup_offsets_are_deterministic() -> None:
    no_history = WatchSpec("carol", "c", 300)
    future = WatchSpec("dave", "d", 300, last_ok=950)
    overdue_b = WatchSpec("bob", "b", 300, last_ok=600)
    overdue_a = WatchSpec("alice", "a", 300, last_ok=600)

    assert initial_watch_delay(no_history, now=1_000) == 300
    assert initial_watch_delay(future, now=1_000) == 250
    assert initial_watch_delay(WatchSpec("rollback", "r", 300, last_ok=1_100), now=1_000) == 400
    assert startup_offsets([overdue_b, future, overdue_a], now=1_000) == {
        "alice": 0.0,
        "bob": 2.0,
        "dave": 250.0,
    }


def test_first_check_is_immediate_only_for_a_clean_new_registration() -> None:
    fresh = WatchSpec("carol", "c", 300)
    failed = WatchSpec("dave", "d", 300, last_error="boom", consecutive_errors=1)
    counted = WatchSpec("erin", "e", 300, consecutive_errors=1)
    recorded = WatchSpec("frank", "f", 300, last_error="boom")

    assert wants_first_check_now(fresh) is True
    assert wants_first_check_now(failed) is False
    assert wants_first_check_now(counted) is False
    assert wants_first_check_now(recorded) is False
    assert wants_first_check_now(WatchSpec("gina", "g", 300, last_ok=900)) is False

    # Daemon role: a clean new row goes now, a failed one waits a full interval.
    assert initial_watch_delay(fresh, now=1_000, first_check_now=True) == 0.0
    assert initial_watch_delay(failed, now=1_000, first_check_now=True) == 300
    assert initial_watch_delay(counted, now=1_000, first_check_now=True) == 300
    assert initial_watch_delay(recorded, now=1_000, first_check_now=True) == 300
    # A recorded success is unaffected by the daemon flag.
    assert initial_watch_delay(WatchSpec("gina", "g", 300, last_ok=950), now=1_000) == 250
    assert (
        initial_watch_delay(
            WatchSpec("gina", "g", 300, last_ok=950), now=1_000, first_check_now=True
        )
        == 250
    )
    # The REPL role keeps today's behaviour exactly.
    assert initial_watch_delay(fresh, now=1_000) == 300
    assert initial_watch_delay(failed, now=1_000) == 300


def test_recovery_staggers_several_never_checked_registrations() -> None:
    specs = [
        WatchSpec("carol", "c", 300),
        WatchSpec("alice", "a", 300),
        WatchSpec("bob", "b", 300),
        WatchSpec("dave", "d", 300, last_error="boom", consecutive_errors=1),
        WatchSpec("erin", "e", 300, last_ok=600),
    ]
    assert startup_offsets(specs, now=1_000, first_check_now=True) == {
        "alice": 0.0,
        "bob": 2.0,
        "carol": 4.0,
        "dave": 300.0,
        "erin": 6.0,
    }
    assert startup_offsets(specs, now=1_000) == {
        "alice": 300.0,
        "bob": 300.0,
        "carol": 300.0,
        "dave": 300.0,
        "erin": 0.0,
    }


def _recording_manager(
    history: HistoryStore, recorded: dict[str, float], *, repl: bool = False
) -> WatchManager:
    manager = _manager(history, repl=repl)
    original = manager.add

    def add(spec: WatchSpec, **kwargs: Any) -> WatchSpec:
        recorded[spec.user] = float(kwargs["initial_delay"])
        return original(spec, **kwargs)

    manager.add = add  # type: ignore[method-assign]
    return manager


async def _reconciled_delays(
    history: HistoryStore, role: WatchExecutorRole, users: list[str]
) -> dict[str, float]:
    recorded: dict[str, float] = {}
    manager = _recording_manager(history, recorded, repl=role == "repl")
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role=role,
        now=lambda: 1_000,
    )
    try:
        await daemon.start()
        # Adding after start exercises the steady-state reconcile path, which is
        # what a user adding their first account in the GUI actually hits.
        for user in users:
            assert history.register_watch(user, 300).spec is not None
        await daemon.reconcile_once()
    finally:
        await daemon.stop()
        if manager.executor_acquired:
            manager.release_executor()
    return recorded


async def test_daemon_checks_a_new_registration_right_away(history: HistoryStore) -> None:
    assert await _reconciled_delays(history, "daemon", ["alice"]) == {"alice": 0.0}


async def test_repl_keeps_the_full_interval_before_a_first_check(history: HistoryStore) -> None:
    assert await _reconciled_delays(history, "repl", ["alice"]) == {"alice": 300.0}


async def test_daemon_keeps_the_interval_for_a_failed_new_registration(
    history: HistoryStore,
) -> None:
    spec = history.register_watch("alice", 300).spec
    assert spec is not None
    assert history.update_watch_state(spec, last_error="boom", consecutive_errors=1)
    recorded: dict[str, float] = {}
    manager = _recording_manager(history, recorded)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
        now=lambda: 1_000,
    )
    try:
        await daemon.start()
    finally:
        await daemon.stop()
        manager.release_executor()
    assert recorded == {"alice": 300.0}


async def test_daemon_recovery_staggers_clean_new_registrations(history: HistoryStore) -> None:
    for user in ("carol", "alice", "bob"):
        assert history.register_watch(user, 300).spec is not None
    recorded: dict[str, float] = {}
    manager = _recording_manager(history, recorded)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
        now=lambda: 1_000,
    )
    try:
        await daemon.start()
    finally:
        await daemon.stop()
        manager.release_executor()
    assert recorded == {"alice": 0.0, "bob": 2.0, "carol": 4.0}


def test_estimate_watch_load_bounds_backend_calls() -> None:
    specs = [WatchSpec("alice", "a", 300), WatchSpec("bob", "b", 600)]
    estimate = estimate_watch_load(specs)
    assert estimate.ticks_per_hour == 18.0
    assert estimate.backend_calls_per_hour_low == 36.0
    assert estimate.backend_calls_per_hour_high == 54.0


async def test_daemon_recovers_active_rows_and_skips_paused(history: HistoryStore) -> None:
    alice = history.register_watch("alice", 300).spec
    bob = history.register_watch("bob", 300).spec
    assert alice is not None and bob is not None
    assert history.update_watch_state(bob, status="paused")
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
        now=lambda: 1_000,
    )

    recovered = await daemon.start()
    assert recovered == 1
    assert [spec.user for spec in manager.list()] == ["alice"]
    assert manager.executor_acquired is True

    await daemon.stop()
    assert manager.executor_acquired is True
    manager.release_executor()


async def test_daemon_start_prunes_expired_and_excess_snapshots(history: HistoryStore) -> None:
    now = int(time.time())
    _seed_snapshot(history, target="expired", captured_at=now - 31 * 86400)
    for offset in range(105):
        _seed_snapshot(history, target="capped", captured_at=now - offset)
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
    )

    await daemon.start()

    assert _snapshot_count(history, "expired") == 0
    assert _snapshot_count(history, "capped") == 100
    await daemon.stop()
    manager.release_executor()


async def test_prune_failures_are_nonfatal_and_periodically_retried(
    history: HistoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def flaky_prune() -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("temporary prune failure")
        return {"cli_history_deleted": 0, "snapshots_deleted": 0}

    monkeypatch.setattr(history, "prune_async", flaky_prune)
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
        prune_seconds=0.01,
    )
    await daemon.start()
    stop = asyncio.Event()
    run_task = asyncio.create_task(daemon.run(stop))

    for _ in range(100):
        if calls >= 3:
            break
        await asyncio.sleep(0.005)

    assert calls >= 3
    assert run_task.done() is False
    stop.set()
    await asyncio.wait_for(run_task, timeout=1)
    manager.release_executor()


async def test_shutdown_drains_started_prune_before_returning(
    history: HistoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    prune_started = threading.Event()
    release_prune = threading.Event()

    def blocking_prune() -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls > 1:
            prune_started.set()
            assert release_prune.wait(timeout=2)
        return {"cli_history_deleted": 0, "snapshots_deleted": 0}

    monkeypatch.setattr(history, "prune", blocking_prune)
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
        prune_seconds=0.01,
    )
    await daemon.start()
    stop = asyncio.Event()
    run_task = asyncio.create_task(daemon.run(stop))
    assert await asyncio.to_thread(prune_started.wait, 1)

    stop.set()
    await asyncio.sleep(0.02)
    assert run_task.done() is False
    release_prune.set()
    await asyncio.wait_for(run_task, timeout=1)
    manager.release_executor()


async def test_repeated_startup_cancellation_drains_prune_before_releasing_executor(
    history: HistoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    prune_started = threading.Event()
    release_prune = threading.Event()
    prune_finished = threading.Event()

    def blocking_prune() -> dict[str, int]:
        prune_started.set()
        assert release_prune.wait(timeout=2)
        prune_finished.set()
        return {"cli_history_deleted": 0, "snapshots_deleted": 0}

    monkeypatch.setattr(history, "prune", blocking_prune)
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
    )
    starting = asyncio.create_task(daemon.start())
    try:
        assert await asyncio.to_thread(prune_started.wait, 1)
        starting.cancel()
        await asyncio.sleep(0)
        starting.cancel()
        done, _ = await asyncio.wait({starting}, timeout=0.02)
        assert not done, "cancelled startup must wait for its SQLite worker"
        assert manager.executor_acquired
        assert not prune_finished.is_set()
    finally:
        release_prune.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, timeout=1)
    assert prune_finished.is_set()
    assert not manager.executor_acquired


async def test_reconcile_add_remove_pause_and_replace(history: HistoryStore) -> None:
    alice = _already_checked(history, "alice")
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
    )
    await daemon.start()

    bob = _already_checked(history, "bob", 600)
    assert bob is not None
    await daemon.reconcile_once()
    assert [spec.user for spec in manager.list()] == ["alice", "bob"]

    assert history.update_watch_state(alice, status="paused")
    await daemon.reconcile_once()
    assert [spec.user for spec in manager.list()] == ["bob"]

    reactivated = history.register_watch("alice", 900).spec
    assert reactivated is not None
    await daemon.reconcile_once()
    current = manager.get("alice")
    assert current is not None
    assert current.registration_id == reactivated.registration_id
    assert current.interval_seconds == 900

    assert history.delete_watch("bob")
    await daemon.reconcile_once()
    assert [spec.user for spec in manager.list()] == ["alice"]
    await daemon.stop()
    manager.release_executor()


async def test_repl_owner_discovers_rows_from_second_store(history: HistoryStore) -> None:
    other = HistoryStore(history.path)
    manager = _manager(history, repl=True)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="repl",
    )
    try:
        assert await daemon.start() == 0
        assert manager.executor_acquired is False
        assert other.register_watch("alice", 300).kind == "created"
        await daemon.reconcile_once()
        assert manager.executor_acquired is True
        assert [spec.user for spec in manager.list()] == ["alice"]

        assert other.delete_watch("alice")
        await daemon.reconcile_once()
        assert manager.list() == []
        assert manager.executor_acquired is False
    finally:
        await daemon.stop()
        other.close()


async def test_repl_stays_control_only_while_daemon_owns_store(history: HistoryStore) -> None:
    assert history.register_watch("alice", 300).spec is not None
    daemon_manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=daemon_manager,
        tick_factory=_ticks([]),
        role="daemon",
    )
    await daemon.start()

    repl_manager = _manager(history, repl=True)
    repl = WatchDaemon(
        history=history,
        manager=repl_manager,
        tick_factory=_ticks([]),
        role="repl",
    )
    try:
        assert await repl.start() == 0
        assert repl.control_only is True
        assert repl_manager.list() == []
    finally:
        await repl.stop()
        await daemon.stop()
        daemon_manager.release_executor()


async def test_due_tick_persists_success_and_stale_callback_stops(history: HistoryStore) -> None:
    original = history.register_watch("alice", 300).spec
    assert original is not None
    original = _persist_state(history, original, last_ok=600)
    calls: list[str] = []
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks(calls),
        role="daemon",
        now=lambda: 1_000,
    )
    await daemon.start()
    persisted = history.get_watch("alice")
    for _ in range(50):
        if calls == ["alice"] and persisted is not None and persisted.last_ok == 1_000:
            break
        await asyncio.sleep(0.01)
        persisted = history.get_watch("alice")
    assert calls == ["alice"]
    assert persisted is not None and persisted.last_ok == 1_000

    assert history.delete_watch("alice")
    replacement = history.register_watch("alice", 600).spec
    assert replacement is not None
    assert history.update_watch_state(original, last_error="stale") is False
    assert history.get_watch("alice") == replacement
    await daemon.stop()
    manager.release_executor()


async def test_failed_tick_is_redacted_in_state_and_executor_output(
    history: HistoryStore,
) -> None:
    secret = "watch-secret-123456"
    register_secret(secret)
    _already_checked(history, "alice")
    messages: list[str] = []

    def failing_tick_factory(user: str) -> TickFn:
        async def tick() -> None:
            raise RuntimeError(f"backend token={secret}")

        return tick

    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=failing_tick_factory,
        role="daemon",
        state_output=messages.append,
    )
    await daemon.start()

    state = await manager.tick_once("alice")

    assert state.consecutive_errors == 1
    persisted = history.get_watch("alice")
    assert persisted is not None
    assert persisted.last_error == "backend token=***"
    assert messages == ["@alice: watch error (1/2) · active · backend token=***"]
    await daemon.stop()
    manager.release_executor()


async def test_state_output_failure_does_not_stop_executor(history: HistoryStore) -> None:
    _already_checked(history, "alice")

    def failing_tick_factory(user: str) -> TickFn:
        async def tick() -> None:
            raise RuntimeError(f"temporary failure for {user}")

        return tick

    def broken_output(_: str) -> None:
        raise RuntimeError("terminal unavailable")

    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=failing_tick_factory,
        role="daemon",
        state_output=broken_output,
    )
    await daemon.start()

    state = await manager.tick_once("alice")

    assert state.status == "active"
    assert state.consecutive_errors == 1
    assert manager.fatal_error.done() is False
    persisted = history.get_watch("alice")
    assert persisted is not None and persisted.consecutive_errors == 1
    await daemon.stop()
    manager.release_executor()


async def test_run_propagates_reconcile_failure_and_drains(
    history: HistoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
        reconcile_seconds=0.01,
    )
    await daemon.start()

    async def broken_list() -> list[WatchSpec]:
        raise RuntimeError("registry failed")

    monkeypatch.setattr(history, "list_watches_async", broken_list)
    with pytest.raises(RuntimeError, match="registry failed"):
        await daemon.run(asyncio.Event())
    assert manager.list() == []
    assert manager.executor_acquired is True
    manager.release_executor()


async def test_start_failure_releases_daemon_lock(
    history: HistoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
    )

    async def broken_list() -> list[WatchSpec]:
        raise RuntimeError("cannot recover")

    monkeypatch.setattr(history, "list_watches_async", broken_list)
    with pytest.raises(RuntimeError, match="cannot recover"):
        await daemon.start()
    assert manager.executor_acquired is False


async def test_run_stops_cleanly_on_event(history: HistoryStore) -> None:
    manager = _manager(history)
    daemon = WatchDaemon(
        history=history,
        manager=manager,
        tick_factory=_ticks([]),
        role="daemon",
        reconcile_seconds=60,
    )
    await daemon.start()
    stop = asyncio.Event()
    task = asyncio.create_task(daemon.run(stop))
    await asyncio.sleep(0)
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    assert manager.executor_acquired is True
    manager.release_executor()
