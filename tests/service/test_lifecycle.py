"""Worker lifecycle with real spawned processes: end-to-end synthetic job and Python/CLI/HTTP parity,
native-thread limits, crashes, descendant cleanup, the commit/cancel order, restart reconciliation
and crash recovery of the publication outbox."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from financeai.application.strategy import StrategyConfig, run_strategy, summarize
from financeai.cli import main as cli_main
from financeai.core.errors import InvalidDataError
from financeai.core.hashing import sha256_file
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.scenario import demo_scenarios
from financeai.domains.market.synthetic import SyntheticMarketSpec, synthetic_market
from financeai.service import coordinator as coordinator_module
from financeai.service.artifacts import tree_digest
from financeai.service.coordinator import Coordinator, _Active
from financeai.service.errors import StateConflictError
from financeai.service.jobs import execute_job, job_from_mapping, request_hash
from financeai.service.models import EXAMPLE_STRATEGY
from financeai.service.settings import ServiceSettings
from financeai.service.store import JobStatus, JobStore, Operation, RunRecord, new_id

from .runners import blocking_runner, child_runner, crash_runner, probe_runner, release_file
from .support import (
    AUTH,
    SettingsFactory,
    get_run,
    post_run,
    run_request,
    service,
    synthetic_job,
    wait_progress,
    wait_status,
)
from .test_jobs import write_caches

BIG_START = "1700000000000000001"  # > 2**53: must survive JSON as an exact string


def test_end_to_end_synthetic_job_matches_python_and_cli(make_settings: SettingsFactory, tmp_path: Path) -> None:
    settings = make_settings()
    spec = SyntheticMarketSpec(seed=7, start_ns=int(BIG_START))
    cfg = StrategyConfig.from_mapping(EXAMPLE_STRATEGY)
    scenario = demo_scenarios()["base"]
    prep = PreparedMarket.build(synthetic_market(spec, order=cfg.instruments))
    python = summarize(run_strategy(prep, cfg, scenario), prep, cfg, scenario)
    caches = write_caches(spec, tmp_path / "caches")
    (tmp_path / "strategy.json").write_text(json.dumps(EXAMPLE_STRATEGY))
    assert (
        cli_main(
            [
                "run",
                str(tmp_path / "strategy.json"),
                "--cache",
                str(caches[0]),
                "--cache",
                str(caches[1]),
                "--out",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    cli_summary = tmp_path / "cli" / "summary.json"

    with service(settings) as client:
        accepted = post_run(client, run_request(synthetic_job(seed=7, start_ns=BIG_START)))
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run_id"]
        done = wait_status(client, run_id, {"SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"})
        assert done["status"] == "SUCCEEDED", done
        result = done["result"]
        assert result["net_pnl"] == python["net_pnl"] == json.loads(cli_summary.read_text())["net_pnl"]
        assert result["round_trips"] == python["round_trips"] > 0
        assert result["first_event_ns"] == str(python["first_event_ns"]) and int(result["first_event_ns"]) > 2**53
        assert done["job"]["start_ns"] == BIG_START and done["evidence_scope"] == ["SYNTHETIC"]
        assert done["economic_status"] == "VALID_ACCOUNT" and done["failure"] is None
        assert done["progress"]["phase"] == "published"
        assert int(done["finished_at_ns"]) >= int(done["started_at_ns"]) >= int(done["created_at_ns"]) > 2**53

        metas: dict[str, list[dict[str, Any]]] = {}
        for artifact_id in done["artifact_ids"]:
            response = client.get(f"/v1/artifacts/{artifact_id}", headers=AUTH)
            assert response.status_code == 200
            metas.setdefault(response.json()["kind"], []).append(response.json())
        assert sorted((k, len(v)) for k, v in metas.items()) == [
            ("fills_csv", 1),
            ("orders_csv", 1),
            ("run_manifest", 1),
            ("run_summary", 1),
            ("strategy_config", 1),
            ("tape_cache", 2),
        ]
        summary_meta = metas["run_summary"][0]
        assert summary_meta["sha256"] == sha256_file(cli_summary)
        assert metas["fills_csv"][0]["sha256"] == sha256_file(tmp_path / "cli" / "fills.csv")
        assert summary_meta["data_role"] == "SYNTHETIC" and summary_meta["run_id"] == run_id
        assert summary_meta["provenance"]["generator"]["start_ns"] == BIG_START
        assert summary_meta["media_type"] == "application/json" and summary_meta["schema_type"]
        assert str(settings.state_root) not in json.dumps(metas)

        # The run's own outputs are registered artifacts: replay them through strategy_replay.
        replay = post_run(
            client,
            run_request(
                {
                    "kind": "strategy_replay",
                    "strategy_artifact_id": metas["strategy_config"][0]["artifact_id"],
                    "cache_artifact_ids": [m["artifact_id"] for m in metas["tape_cache"]],
                    "scenario": "base",
                }
            ),
            key="k2",
        )
        assert replay.status_code == 202, replay.text
        second = wait_status(client, replay.json()["run_id"], {"SUCCEEDED", "FAILED"})
        assert second["status"] == "SUCCEEDED", second
        assert second["result"]["net_pnl"] == python["net_pnl"] and second["evidence_scope"] == ["SYNTHETIC"]

    committed = settings.artifacts_dir / "runs" / run_id / done["attempt_id"]
    manifest = json.loads((committed / "manifest.json").read_text())
    assert manifest["run_id"] == run_id and len(manifest["artifacts"]) == 6
    for entry in manifest["artifacts"]:
        assert tree_digest(committed / entry["path"]) == (entry["sha256"], entry["size_bytes"])
    assert not any(settings.staging_dir.iterdir())


def test_workers_bound_native_threads_and_lead_their_process_group(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    with service(settings, probe_runner) as client:
        run_id = post_run(client, run_request(synthetic_job())).json()["run_id"]
        assert wait_status(client, run_id, {"SUCCEEDED", "FAILED"})["status"] == "SUCCEEDED"
    probe = json.loads((settings.state_root / f"probe-{run_id}.json").read_text())
    assert set(probe["thread_env"].values()) == {"1"}
    assert probe["numpy_loaded_before_job"] is False and probe["fastapi_loaded"] is False
    assert probe["process_group_leader"] is True and probe["pid"] != os.getpid()


def test_a_crashed_worker_fails_the_run_and_the_next_job_is_healthy(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    with service(settings, crash_runner) as client:
        run_id = post_run(client, run_request(synthetic_job())).json()["run_id"]
        failed = wait_status(client, run_id, {"FAILED", "SUCCEEDED"})
        assert failed["status"] == "FAILED" and failed["failure"]["code"] == "worker_crashed"
        assert failed["artifact_ids"] == [] and failed["economic_status"] == "INCOMPLETE"
    assert list((settings.quarantine_dir / run_id).glob("*-crashed-*/partial.txt"))
    with service(settings) as client:
        healthy = post_run(client, run_request(synthetic_job()), key="next").json()["run_id"]
        assert wait_status(client, healthy, {"SUCCEEDED", "FAILED"})["status"] == "SUCCEEDED"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_cancelling_a_job_kills_its_descendants(make_settings: SettingsFactory) -> None:
    settings = make_settings(cancel_grace_s=0.5)
    with service(settings, child_runner) as client:
        run_id = post_run(client, run_request(synthetic_job())).json()["run_id"]
        pid_file = settings.state_root / f"child-{run_id}.pid"
        deadline = time.monotonic() + 120
        while not pid_file.exists() or not pid_file.read_text():
            assert time.monotonic() < deadline, "child never started"
            time.sleep(0.05)
        child = int(pid_file.read_text())
        assert _alive(child)
        assert client.post(f"/v1/runs/{run_id}/cancel", headers=AUTH).status_code == 202
        done = wait_status(client, run_id, {"CANCELLED"}, timeout_s=60)
        assert done["failure"]["code"] == "terminated"
        deadline = time.monotonic() + 30
        while _alive(child):
            assert time.monotonic() < deadline, "descendant survived its worker"
            time.sleep(0.05)


def _attempt(
    settings: ServiceSettings, key: str = "k", coordinator: Coordinator | None = None
) -> tuple[Coordinator, RunRecord, Path]:
    """A coordinator without its dispatch thread and one RUNNING attempt (outputs not yet staged)."""
    coordinator = coordinator or Coordinator(settings)
    job = job_from_mapping(synthetic_job())
    record, _ = coordinator.store.admit(
        campaign_id="c",
        idempotency_key=key,
        request_hash=request_hash("c", job),
        request=job.to_mapping(),
        kind=job.kind,
        evidence_scope=("SYNTHETIC",),
        queue_capacity=10,
    )
    running = coordinator.store.start_attempt(record.run_id)
    assert running.attempt_id is not None
    staging = settings.staging_dir / running.run_id / running.attempt_id
    staging.mkdir(parents=True)
    return coordinator, running, staging


def _completed(running: RunRecord, staging: Path) -> tuple[str, dict[str, Any]]:
    return "completed", dict(execute_job({"job": dict(running.request)}, staging, lambda *_: None))


def test_completion_before_cancellation_wins_and_late_output_is_rejected(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    coordinator, running, staging = _attempt(settings)
    assert running.attempt_id is not None
    outcome = _completed(running, staging)
    coordinator.store.request_cancel(running.run_id)  # a cancel request racing the completion
    done = coordinator.settle_attempt(running.run_id, running.attempt_id, outcome, staging)
    assert done.status is JobStatus.SUCCEEDED and len(coordinator.store.artifacts_for_run(running.run_id)) == 7
    # A duplicate/late delivery of the same completion cannot publish twice.
    late = coordinator.settle_attempt(running.run_id, running.attempt_id, outcome, staging)
    assert late.status is JobStatus.SUCCEEDED and len(coordinator.store.artifacts_for_run(running.run_id)) == 7

    _, run2, staging2 = _attempt(settings, "k2", coordinator)
    assert run2.attempt_id is not None
    outcome2 = _completed(run2, staging2)
    coordinator.store.request_cancel(run2.run_id)
    cancelled = coordinator.settle_attempt(run2.run_id, run2.attempt_id, outcome2, staging2, invalidated=True)
    assert cancelled.status is JobStatus.CANCELLED and coordinator.store.artifacts_for_run(run2.run_id) == []
    assert not staging2.exists() and list((settings.quarantine_dir / run2.run_id).glob("*-completed-*"))
    late_dir = settings.staging_dir / "late"
    late_dir.mkdir(parents=True)
    rejected = coordinator.settle_attempt(run2.run_id, run2.attempt_id, outcome2, late_dir)
    assert rejected.status is JobStatus.CANCELLED and coordinator.store.artifacts_for_run(run2.run_id) == []
    assert not late_dir.exists() and list((settings.quarantine_dir / run2.run_id).glob("*-stale-*"))
    coordinator.stop()


def test_output_not_matching_the_descriptor_is_never_published(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    coordinator, running, staging = _attempt(settings)
    assert running.attempt_id is not None
    kind, descriptor = _completed(running, staging)
    (staging / "run" / "fills.csv").write_text("forged\n")
    failed = coordinator.settle_attempt(running.run_id, running.attempt_id, (kind, descriptor), staging)
    assert failed.status is JobStatus.FAILED and failed.failure is not None
    assert failed.failure["code"] == "invalid_data" and coordinator.store.artifacts_for_run(running.run_id) == []
    coordinator.stop()


def _within(seconds: float, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` on a daemon thread and fail (instead of hanging the suite) if it has not returned in time."""
    box: list[Any] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            box.append(fn())
        except BaseException as exc:  # re-raised on the test thread
            errors.append(exc)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "the call did not return in time"
    if errors:
        raise errors[0]
    return box[0]


def test_special_files_in_staging_are_quarantined_instead_of_blocking_the_coordinator(
    make_settings: SettingsFactory,
) -> None:
    """Regression: the stray-file check saw only regular files, so a FIFO in the attempt directory passed it
    and the fsync pass then opened it for reading, blocking forever under the coordinator lock (dispatch,
    cancellation and polling all stalled). Special files and symlinks are rejected like unexpected files."""
    settings = make_settings()
    coordinator, running, staging = _attempt(settings)
    assert running.attempt_id is not None
    attempt = running.attempt_id
    outcome = _completed(running, staging)
    os.mkfifo(staging / "run" / "stray.fifo")
    failed = _within(10, lambda: coordinator.settle_attempt(running.run_id, attempt, outcome, staging))
    assert failed.status is JobStatus.FAILED and failed.failure is not None
    assert failed.failure["code"] == "invalid_data" and coordinator.store.artifacts_for_run(running.run_id) == []
    assert not staging.exists() and list((settings.quarantine_dir / running.run_id).glob("*-rejected-*/run/stray.fifo"))

    _, run2, staging2 = _attempt(settings, "k2", coordinator)
    assert run2.attempt_id is not None
    attempt2 = run2.attempt_id
    outcome2 = _completed(run2, staging2)
    (staging2 / "run" / "alias.csv").symlink_to(staging2 / "run" / "fills.csv")
    failed2 = _within(10, lambda: coordinator.settle_attempt(run2.run_id, attempt2, outcome2, staging2))
    assert failed2.status is JobStatus.FAILED and coordinator.store.artifacts_for_run(run2.run_id) == []

    fifo = settings.staging_dir / "lone.fifo"
    os.mkfifo(fifo)
    with pytest.raises(InvalidDataError):  # the fsync itself refuses to open anything but files and directories
        _within(10, lambda: coordinator_module._fsync_path(fifo))
    coordinator.stop()


def test_a_dispatch_call_does_bounded_work_under_the_lock(make_settings: SettingsFactory) -> None:
    """Regression: _dispatch looped under the coordinator lock until nothing dispatchable was queued, so a
    backlog of runs failing at input resolution (each a synchronous=FULL write) stalled settlement, cancellation
    and polling for the whole backlog. One call attempts at most a few launches per worker slot and wakes the
    next cycle to continue."""
    settings = make_settings(max_workers=1, queue_capacity=1000)
    coordinator = Coordinator(settings)  # no dispatch thread: _dispatch is driven by hand
    unregistered = {
        "kind": "strategy_replay",
        "strategy_artifact_id": new_id("art"),
        "cache_artifact_ids": [new_id("art")],
        "scenario": "base",
    }
    job = job_from_mapping(unregistered)
    for i in range(20):
        coordinator.store.admit(
            campaign_id=f"c{i}",
            idempotency_key="k",
            request_hash=request_hash(f"c{i}", job),
            request=job.to_mapping(),
            kind=job.kind,
            evidence_scope=("SYNTHETIC",),
            queue_capacity=1000,
        )
    coordinator._wake.clear()
    coordinator._dispatch()
    failed = coordinator.store.runs_with_status(JobStatus.FAILED)
    assert len(failed) == coordinator_module.DISPATCH_ATTEMPTS_PER_SLOT * settings.max_workers < 20
    assert all(r.failure is not None and r.failure["code"] == "not_found" for r in failed)
    assert len(coordinator.store.runs_with_status(JobStatus.QUEUED)) == 20 - len(failed)
    assert coordinator._wake.is_set()  # the backlog is finished by the following cycles, not by this call
    for _ in range(10):
        coordinator._dispatch()
    assert coordinator.store.runs_with_status(JobStatus.QUEUED) == []
    coordinator.store.close()


def test_restart_interrupts_active_runs_and_keeps_the_queue(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    coordinator, running, staging = _attempt(settings)
    (staging / "partial.txt").write_text("half")
    job = job_from_mapping(synthetic_job(seed=9))
    queued, _ = coordinator.store.admit(
        campaign_id="c",
        idempotency_key="queued",
        request_hash=request_hash("c", job),
        request=job.to_mapping(),
        kind=job.kind,
        evidence_scope=("SYNTHETIC",),
        queue_capacity=10,
    )
    coordinator.store.close()  # the process "dies": no shutdown handling at all

    with service(settings, blocking_runner) as client:
        interrupted = get_run(client, running.run_id)
        assert interrupted["status"] == "INTERRUPTED" and interrupted["failure"]["code"] == "interrupted"
        assert interrupted["generation"] == 1 and interrupted["attempt_id"] == running.attempt_id
        assert interrupted["economic_status"] == "INCOMPLETE" and interrupted["artifact_ids"] == []
        assert list((settings.quarantine_dir / running.run_id).glob("*-interrupted-*/partial.txt"))
        restarted = wait_progress(client, queued.run_id, "blocked")
        assert restarted["status"] == "RUNNING"
        assert client.post(f"/v1/runs/{running.run_id}/cancel", headers=AUTH).status_code == 200
        release_file(settings.state_root, queued.run_id).touch()
        assert wait_status(client, queued.run_id, {"SUCCEEDED", "FAILED"})["status"] == "SUCCEEDED"
    with service(settings, blocking_runner) as client:
        assert get_run(client, running.run_id)["status"] == "INTERRUPTED"  # never resubmitted


def test_graceful_shutdown_interrupts_running_jobs(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    with service(settings, blocking_runner) as client:
        run_id = post_run(client, run_request(synthetic_job())).json()["run_id"]
        wait_progress(client, run_id, "blocked")
    store = JobStore(settings.db_path)
    record = store.get_run(run_id)
    store.close()
    assert record.status is JobStatus.INTERRUPTED and record.failure is not None
    assert record.failure["code"] == "service_shutdown"


def test_one_coordinator_owns_a_state_root(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    first = Coordinator(settings)
    first.start()
    second = Coordinator(settings)
    with pytest.raises(StateConflictError):
        second.start()  # its recovery would otherwise interrupt the first owner's live runs
    with pytest.raises(StateConflictError):
        second.recover()
    assert (first.available, second.available) == (True, False)
    first.stop()
    second.start()
    assert second.available
    second.stop()


def test_dispatch_is_not_blocked_by_one_campaign_s_backlog(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the dispatcher scanned a fixed window of the oldest queued runs, so about 66 queued runs of a
    busy campaign hid every other campaign's run and a free worker slot stayed idle (head-of-line blocking)."""
    settings = make_settings(max_workers=2, queue_capacity=1000)
    coordinator = Coordinator(settings)  # no dispatch thread: _dispatch is driven by hand
    job = job_from_mapping(synthetic_job())
    for campaign, key in [("A", f"a-{i}") for i in range(72)] + [("B", "b-0")]:
        coordinator.store.admit(
            campaign_id=campaign,
            idempotency_key=key,
            request_hash=request_hash(campaign, job),
            request=job.to_mapping(),
            kind=job.kind,
            evidence_scope=("SYNTHETIC",),
            queue_capacity=1000,
        )
    ctx = multiprocessing.get_context("spawn")
    launched: list[str] = []

    def fake_launch(record: RunRecord) -> bool:  # occupies a worker slot without starting a process
        launched.append(record.campaign_id)
        receiver, sender = ctx.Pipe(duplex=False)
        sender.close()
        coordinator._active[record.run_id] = _Active(
            run_id=record.run_id,
            campaign_id=record.campaign_id,
            attempt_id="att",
            process=ctx.Process(),
            conn=receiver,
            cancel_event=ctx.Event(),
            staging=settings.staging_dir,
        )
        return True

    monkeypatch.setattr(coordinator, "_launch", fake_launch)
    coordinator._dispatch()
    assert launched == ["A", "B"]  # the oldest A run, then B's run from behind A's backlog
    coordinator._dispatch()
    assert launched == ["A", "B"]  # both slots busy: nothing more
    for active in coordinator._active.values():
        active.conn.close()
    coordinator._active.clear()
    coordinator.store.close()


def test_staged_output_is_synced_before_the_publication_is_recorded(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: only manifest.json was fsynced, so after a power loss a SUCCEEDED run (the database is
    synchronous=FULL) could reference run files and cache columns whose bytes never reached the disk."""
    settings = make_settings()
    coordinator, running, staging = _attempt(settings)
    assert running.attempt_id is not None
    outcome = _completed(running, staging)
    staged = {p for p in staging.rglob("*") if p.is_file()}
    assert len(staged) >= 6
    synced: list[Path] = []
    real_fsync = coordinator_module._fsync_path

    def recording_fsync(path: Path) -> None:
        synced.append(Path(path))
        real_fsync(path)

    monkeypatch.setattr(coordinator_module, "_fsync_path", recording_fsync)
    at_begin: set[Path] = set()
    real_begin = JobStore.begin_publication

    def spy(self: JobStore, *args: Any, **kwargs: Any) -> Operation:
        at_begin.update(synced)
        return real_begin(self, *args, **kwargs)

    monkeypatch.setattr(JobStore, "begin_publication", spy)
    done = coordinator.settle_attempt(running.run_id, running.attempt_id, outcome, staging)
    assert done.status is JobStatus.SUCCEEDED
    assert staged | {staging / "manifest.json"} <= at_begin  # every byte the commit will reference
    assert {staging, staging / "run"} <= at_begin  # and the directory entries that name them
    coordinator.store.close()


class SimulatedCrash(Exception):
    pass


@pytest.mark.parametrize("crash_point", ["after_rename", "before_rename"])
def test_publication_outbox_recovers_exactly_once(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch, crash_point: str
) -> None:
    settings = make_settings()
    coordinator, running, staging = _attempt(settings)
    assert running.attempt_id is not None
    outcome = _completed(running, staging)

    def crash(*args: Any, **kwargs: Any) -> None:
        raise SimulatedCrash

    if crash_point == "after_rename":
        monkeypatch.setattr(JobStore, "complete_publication", crash)
    else:
        monkeypatch.setattr(Coordinator, "_move_into_place", crash)
    with pytest.raises(SimulatedCrash):
        coordinator.settle_attempt(running.run_id, running.attempt_id, outcome, staging)
    assert [o.run_id for o in coordinator.store.pending_operations()] == [running.run_id]
    coordinator.store.close()
    monkeypatch.undo()

    recovered = Coordinator(settings)
    recovered.recover()
    recovered.recover()  # idempotent
    record = recovered.store.get_run(running.run_id)
    artifacts = recovered.store.artifacts_for_run(running.run_id)
    assert recovered.store.pending_operations() == []
    if crash_point == "after_rename":
        assert record.status is JobStatus.SUCCEEDED and len(artifacts) == 7
        assert record.result is not None and record.result["account_valid"] is True
        for artifact in artifacts:
            recovered.artifacts.verify(artifact)
    else:
        assert record.status is JobStatus.INTERRUPTED and artifacts == []
        assert not (settings.artifacts_dir / "runs" / running.run_id).exists()
        assert list((settings.quarantine_dir / running.run_id).glob("*-interrupted-*/run/summary.json"))
    recovered.stop()
