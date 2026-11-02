"""Job commands: validation, normalization and idempotency hashes, synchronous execution, Python/CLI
parity, content digests and artifact registration (synthetic data only)."""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from financeai.application.strategy import StrategyConfig, run_strategy, summarize
from financeai.cli import main as cli_main
from financeai.core.errors import IncompatibleSchemaError, InvalidConfigError, InvalidDataError
from financeai.core.hashing import sha256_file
from financeai.domains.market.cache import save_tapes
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.scenario import demo_scenarios
from financeai.domains.market.synthetic import (
    SyntheticMarketSpec,
    max_span_ns,
    synthetic_market,
    synthetic_source,
    synthetic_tapes,
)
from financeai.service import artifacts as artifacts_module
from financeai.service.artifacts import ArtifactStore, register_strategy, register_tape_cache, tree_digest
from financeai.service.jobs import (
    RUN_OUTPUTS,
    StrategyReplayJob,
    SyntheticReplayJob,
    execute_job,
    expected_outputs,
    job_from_mapping,
    request_hash,
    result_from_summary,
    run_strategy_replay,
    run_synthetic_replay,
    venue_dirname,
)
from financeai.service.models import EXAMPLE_STRATEGY
from financeai.service.store import JobStore, new_id

from .support import synthetic_job


def write_caches(spec: SyntheticMarketSpec, root: Path) -> list[Path]:
    tapes = synthetic_tapes(spec)
    dirs = []
    for key in spec.venues:
        target = root / venue_dirname(key)
        save_tapes(tapes[key], target, source=synthetic_source(spec, key))
        dirs.append(target)
    return dirs


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[tuple[JobStore, ArtifactStore]]:
    store = JobStore(tmp_path / "state" / "service.sqlite3")
    yield store, ArtifactStore(tmp_path / "state" / "artifacts")
    store.close()


def test_equivalent_requests_normalize_to_one_hash() -> None:
    a = job_from_mapping(synthetic_job())
    reordered_strategy = dict(reversed(list(EXAMPLE_STRATEGY.items())))
    b = job_from_mapping(
        synthetic_job(
            venues=list(reversed(synthetic_job()["venues"])),
            rows_per_venue=300,
            start_ns=0,
            strategy=reordered_strategy,
        )
    )
    assert isinstance(a, SyntheticReplayJob)
    assert a.to_mapping() == b.to_mapping()
    assert a.to_mapping()["venues"] == ["venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"]
    assert request_hash("c", a) == request_hash("c", b)
    assert request_hash("c", a) != request_hash("other", a)
    assert request_hash("c", a) != request_hash("c", job_from_mapping(synthetic_job(seed=8)))
    assert job_from_mapping(a.to_mapping()) == a  # the normalized form is itself a valid command
    assert a.evidence_scope == ("SYNTHETIC",)


def test_strategy_replay_cache_ids_are_normalized() -> None:
    ids = [new_id("art") for _ in range(3)]
    job = job_from_mapping(
        {
            "kind": "strategy_replay",
            "strategy_artifact_id": ids[0],
            "cache_artifact_ids": [ids[2], ids[1]],
            "scenario": "base",
        }
    )
    assert isinstance(job, StrategyReplayJob)
    assert job.cache_artifact_ids == tuple(sorted(ids[1:]))
    swapped = StrategyReplayJob(ids[0], (ids[1], ids[2]), "base")
    assert request_hash("c", job) == request_hash("c", swapped)


def _strategy(**changes: Any) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(json.dumps(EXAMPLE_STRATEGY))
    doc.update(changes)
    return doc


@pytest.mark.parametrize(
    ("doc", "error"),
    [
        ({"kind": "train_model"}, InvalidConfigError),
        ({**synthetic_job(), "extra": 1}, InvalidConfigError),
        (synthetic_job(scenario="moon"), InvalidConfigError),
        (synthetic_job(venues=["venue_a/ALPHA-USD"]), InvalidConfigError),  # strategy also trades venue_b
        (synthetic_job(venues="venue_a/ALPHA-USD"), InvalidConfigError),
        (synthetic_job(strategy=_strategy(extra=1)), InvalidConfigError),
        (synthetic_job(strategy=_strategy(schema="other")), IncompatibleSchemaError),
        (synthetic_job(strategy=_strategy(schema_version="2.0")), IncompatibleSchemaError),
        (synthetic_job(seed=-1), InvalidConfigError),
        (synthetic_job(start_ns="0"), InvalidConfigError),
        (
            {
                "kind": "strategy_replay",
                "strategy_artifact_id": "../strategy.json",
                "cache_artifact_ids": [new_id("art")],
                "scenario": "base",
            },
            InvalidConfigError,
        ),
        (
            {
                "kind": "strategy_replay",
                "strategy_artifact_id": new_id("art"),
                "cache_artifact_ids": [],
                "scenario": "base",
            },
            InvalidConfigError,
        ),
        (
            {
                "kind": "strategy_replay",
                "strategy_artifact_id": new_id("art"),
                "cache_artifact_ids": ["art_x"],
                "scenario": "base",
            },
            InvalidConfigError,
        ),
        (
            {
                "kind": "strategy_replay",
                "strategy_artifact_id": new_id("art"),
                "cache_artifact_ids": ["art_" + "1" * 32, "art_" + "1" * 32],
                "scenario": "base",
            },
            InvalidConfigError,
        ),
        (
            {
                "kind": "strategy_replay",
                "strategy_artifact_id": new_id("art"),
                "cache_artifact_ids": [new_id("art")],
                "scenario": "base",
                "path": "/etc/passwd",
            },
            InvalidConfigError,
        ),
    ],
)
def test_invalid_job_commands_are_rejected(doc: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        job_from_mapping(doc)


def test_python_cli_and_job_paths_agree_exactly(tmp_path: Path) -> None:
    job = job_from_mapping(synthetic_job(seed=7))
    assert isinstance(job, SyntheticReplayJob)
    cfg = StrategyConfig.from_mapping(EXAMPLE_STRATEGY)
    scenario = demo_scenarios()["base"]
    # Python: application layer on an in-memory synthetic market.
    prep = PreparedMarket.build(synthetic_market(job.market, order=cfg.instruments))
    python = summarize(run_strategy(prep, cfg, scenario), prep, cfg, scenario)
    # CLI: the same market as immutable caches, then `financeai run`.
    caches = write_caches(job.market, tmp_path / "caches")
    cfg_path = tmp_path / "strategy.json"
    cfg_path.write_text(json.dumps(EXAMPLE_STRATEGY))
    args = [
        "run",
        str(cfg_path),
        *(a for c in caches for a in ("--cache", str(c))),
        "--scenario",
        "base",
        "--out",
        str(tmp_path / "cli"),
    ]
    assert cli_main(args) == 0
    cli = json.loads((tmp_path / "cli" / "summary.json").read_text())
    # Job command (what the HTTP worker executes).
    service = run_synthetic_replay(job, tmp_path / "svc")
    assert python["valid"] and python["round_trips"] > 0 and python["net_pnl"] != 0.0
    assert python["net_pnl"] == cli["net_pnl"] == service["net_pnl"]
    assert python["round_trips"] == cli["round_trips"] == service["round_trips"]
    for name in ("summary.json", "fills.csv", "orders.csv"):
        assert sha256_file(tmp_path / "cli" / name) == sha256_file(tmp_path / "svc" / "run" / name), name


def test_execute_job_reports_digests_and_checkpoints(tmp_path: Path) -> None:
    phases: list[str] = []
    job = synthetic_job(seed=3)
    staging = tmp_path / "attempt"
    staging.mkdir()
    descriptor = execute_job({"job": job}, staging, lambda phase, done, total: phases.append(phase))
    assert phases == ["generate", "load", "replay", "write"]
    parsed = job_from_mapping(job)
    assert [(o["rel_path"], o["kind"]) for o in descriptor["outputs"]] == list(expected_outputs(parsed))
    for output in descriptor["outputs"]:
        assert (output["sha256"], output["size_bytes"]) == tree_digest(staging / output["rel_path"])
    assert {p for p, _ in RUN_OUTPUTS} <= {o["rel_path"] for o in descriptor["outputs"]}


def test_a_raising_checkpoint_stops_before_outputs(tmp_path: Path) -> None:
    class Stop(Exception):
        pass

    def checkpoint(phase: str, done: int, total: int) -> None:
        if phase == "replay":
            raise Stop

    job = job_from_mapping(synthetic_job(seed=3, rows_per_venue=50))
    assert isinstance(job, SyntheticReplayJob)
    with pytest.raises(Stop):
        run_synthetic_replay(job, tmp_path / "out", checkpoint=checkpoint)
    assert (tmp_path / "out" / "caches").is_dir() and not (tmp_path / "out" / "run").exists()
    with pytest.raises(InvalidConfigError):  # outputs never overwrite an existing attempt
        run_synthetic_replay(job, tmp_path / "out")


START_CAP = 2**62


def _clocks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_replay_at_the_latest_start_clock_is_an_exact_time_shift(tmp_path: Path) -> None:
    """Regression: near the int64 ceiling arrival/ack clocks wrapped negative (and wrapped arrivals slipped past
    the execution cutoff) while the run still SUCCEEDED. At the cap the replay must equal the one at 0, shifted."""

    def run(start_ns: int, out: Path) -> dict[str, Any]:
        job = job_from_mapping(synthetic_job(seed=1, start_ns=start_ns, scenario="longer_delay"))
        assert isinstance(job, SyntheticReplayJob)
        return run_synthetic_replay(job, out)

    base, top = run(0, tmp_path / "zero"), run(START_CAP, tmp_path / "top")
    assert top["orders"] == base["orders"] > 0 and top["fills"] == base["fills"]
    assert top["valid"] == base["valid"] and top["net_pnl"] == base["net_pnl"]
    for name, clocks in (
        ("orders.csv", ("submit_ns", "arrival_ns", "ack_ns")),
        ("fills.csv", ("submit_ns", "arrival_ns", "ack_ns")),
    ):
        shifted = _clocks(tmp_path / "top" / "run" / name)
        for a, b in zip(_clocks(tmp_path / "zero" / "run" / name), shifted, strict=True):
            assert [int(b[c]) - int(a[c]) for c in clocks] == [START_CAP] * len(clocks)
            assert START_CAP <= int(b["submit_ns"]) <= int(b["arrival_ns"]) <= int(b["ack_ns"])


def test_the_overflowing_start_clock_is_rejected_before_any_work() -> None:
    with pytest.raises(InvalidConfigError):
        job_from_mapping(
            synthetic_job(seed=1, rows_per_venue=10, start_ns=2**63 - 1 - max_span_ns(10), scenario="longer_delay")
        )


def test_replay_refuses_market_clocks_without_int64_headroom(tmp_path: Path) -> None:
    """A registered cache may carry any int64 clocks: the job path refuses instead of wrapping them."""
    spec = SyntheticMarketSpec(seed=1)
    tapes = synthetic_tapes(spec)
    shift = 2**63 - 1 - max(int(t.book.recv_ns[-1]) for t in tapes.values()) - 10**6  # last event 1 ms below
    caches = []
    for key, t in tapes.items():
        book = dataclasses.replace(t.book, recv_ns=t.book.recv_ns + shift, exch_ns=t.book.exch_ns + shift)
        caches.append(tmp_path / venue_dirname(key))
        save_tapes(dataclasses.replace(t, book=book), caches[-1], source=synthetic_source(spec, key))
    strategy = tmp_path / "strategy.json"
    strategy.write_text(json.dumps(EXAMPLE_STRATEGY))
    with pytest.raises(InvalidDataError, match="int64"):
        run_strategy_replay(strategy, caches, "base", tmp_path / "out")
    assert not (tmp_path / "out" / "run").exists()


def test_tree_digest_covers_names_sizes_and_bytes(tmp_path: Path) -> None:
    (tmp_path / "d" / "sub").mkdir(parents=True)
    (tmp_path / "d" / "a.txt").write_text("alpha")
    (tmp_path / "d" / "sub" / "b.txt").write_text("beta")
    digest, size = tree_digest(tmp_path / "d")
    assert size == 9 and len(digest) == 64
    assert tree_digest(tmp_path / "d" / "a.txt") == (sha256_file(tmp_path / "d" / "a.txt"), 5)
    (tmp_path / "d" / "sub" / "b.txt").write_text("BETA")
    assert tree_digest(tmp_path / "d")[0] != digest
    os.symlink(tmp_path / "d" / "a.txt", tmp_path / "d" / "link.txt")
    with pytest.raises(InvalidDataError):
        tree_digest(tmp_path / "d")
    with pytest.raises(InvalidDataError):
        tree_digest(tmp_path / "missing")


def test_register_strategy_keeps_exact_bytes_and_detects_tampering(
    tmp_path: Path, registry: tuple[JobStore, ArtifactStore]
) -> None:
    store, artifacts = registry
    source = tmp_path / "my-strategy.json"
    source.write_text(json.dumps(EXAMPLE_STRATEGY, indent=1))
    record = register_strategy(store, artifacts, source)
    assert record.kind == "strategy_config" and record.data_role == "NONE" and record.run_id is None
    assert record.provenance["source_name"] == "my-strategy.json"
    assert record.provenance["instruments"] == EXAMPLE_STRATEGY["instruments"]
    path = artifacts.verify(record)
    assert path.read_bytes() == source.read_bytes() and path.is_relative_to(artifacts.root)
    assert store.get_artifact(record.artifact_id) == record
    os.chmod(path, 0o644)  # the stored copy is read-only: tampering means overriding that first
    path.write_text(path.read_text() + " ")
    with pytest.raises(InvalidDataError):
        artifacts.verify(record)
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema": "financeai.strategy", "schema": "x"}')
    with pytest.raises(InvalidConfigError):
        register_strategy(store, artifacts, bad)


def test_register_tape_cache_verifies_columns_and_roles(
    tmp_path: Path, registry: tuple[JobStore, ArtifactStore]
) -> None:
    store, artifacts = registry
    caches = write_caches(SyntheticMarketSpec(seed=2, rows_per_venue=20), tmp_path / "caches")
    record = register_tape_cache(store, artifacts, caches[0], data_role="DEVELOPMENT")
    assert record.kind == "tape_cache" and record.data_role == "DEVELOPMENT"
    assert record.provenance["instrument"] == "venue_a/ALPHA-USD"
    assert artifacts.verify(record).is_dir()
    assert (record.sha256, record.size_bytes) == tree_digest(caches[0])
    with pytest.raises(InvalidConfigError):
        register_tape_cache(store, artifacts, caches[1], data_role="TEST")
    column = caches[1] / "book.bid_px.npy"
    arr = np.load(column)
    arr[0, 0] += 1
    os.chmod(column, 0o644)
    np.save(column, arr)
    with pytest.raises(InvalidDataError):
        register_tape_cache(store, artifacts, caches[1], data_role="SYNTHETIC")
    assert [p.name for p in (artifacts.root / "inputs").iterdir()] == [record.artifact_id]


def test_registered_inputs_are_private_read_only_copies(
    tmp_path: Path, registry: tuple[JobStore, ArtifactStore]
) -> None:
    """Regression: registrations hard-linked the caller's files, so editing the source changed (and corrupted)
    the registered version while GET /v1/artifacts still reported the old digest."""
    store, artifacts = registry
    caches = write_caches(SyntheticMarketSpec(seed=3, rows_per_venue=20), tmp_path / "caches")
    record = register_tape_cache(store, artifacts, caches[0], data_role="DEVELOPMENT")
    stored = artifacts.verify(record)
    sources = sorted(p for p in caches[0].rglob("*") if p.is_file())
    assert sources
    for src in sources:
        info = (stored / src.relative_to(caches[0])).stat()
        assert info.st_nlink == 1 and info.st_ino != src.stat().st_ino, src.name
        assert info.st_mode & 0o222 == 0, src.name  # read-only for everyone
    column = caches[0] / "book.bid_sz.npy"
    os.chmod(column, 0o644)
    data = bytearray(column.read_bytes())
    data[-1] ^= 0xFF
    column.write_bytes(bytes(data))
    assert artifacts.verify(record) == stored  # the registered version is untouched by the source edit
    source = tmp_path / "strategy.json"
    source.write_text(json.dumps(EXAMPLE_STRATEGY))
    strategy = artifacts.verify(register_strategy(store, artifacts, source))
    assert strategy.stat().st_nlink == 1 and strategy.stat().st_mode & 0o222 == 0


def test_a_tape_cache_rewritten_during_registration_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: tuple[JobStore, ArtifactStore]
) -> None:
    """Regression: the source was verified, then copied, and only the copy was hashed, so a cache rewritten
    between the verification and the copy registered a torn tree whose recorded digest was self-consistent
    (every later digest check passed while the column hashes no longer matched the manifest)."""
    store, artifacts = registry
    caches = write_caches(SyntheticMarketSpec(seed=5, rows_per_venue=20), tmp_path / "caches")
    real_copy = artifacts_module._copy_read_only
    rewritten: list[str] = []

    def racing_copy(src: str, dst: str) -> None:
        if src.endswith(".npy") and not rewritten:  # the caller rewrites one column just before it is copied
            rewritten.append(src)
            data = bytearray(Path(src).read_bytes())
            data[-1] ^= 0xFF
            Path(src).write_bytes(bytes(data))
        real_copy(src, dst)

    monkeypatch.setattr(artifacts_module, "_copy_read_only", racing_copy)
    with pytest.raises(InvalidDataError, match="changed during registration"):
        register_tape_cache(store, artifacts, caches[0], data_role="DEVELOPMENT")
    assert rewritten
    assert not (artifacts.root / "inputs").exists() or list((artifacts.root / "inputs").iterdir()) == []
    assert list((artifacts.root / ".incoming").iterdir()) == []
    assert store._conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0  # nothing registered


def test_strategy_replay_verifies_registered_inputs(tmp_path: Path, registry: tuple[JobStore, ArtifactStore]) -> None:
    store, artifacts = registry
    spec = SyntheticMarketSpec(seed=7)
    caches = [
        register_tape_cache(store, artifacts, d, data_role="SYNTHETIC") for d in write_caches(spec, tmp_path / "caches")
    ]
    source = tmp_path / "strategy.json"
    source.write_text(json.dumps(EXAMPLE_STRATEGY))
    strategy = register_strategy(store, artifacts, source)
    job = {
        "kind": "strategy_replay",
        "strategy_artifact_id": strategy.artifact_id,
        "cache_artifact_ids": [c.artifact_id for c in caches],
        "scenario": "base",
    }

    def entry(a: Any) -> dict[str, Any]:
        return {
            "id": a.artifact_id,
            "path": str(artifacts.path_of(a.rel_path)),
            "sha256": a.sha256,
            "size_bytes": a.size_bytes,
        }

    inputs = {"strategy": entry(strategy), "caches": [entry(c) for c in caches]}
    staging = tmp_path / "attempt"
    staging.mkdir()
    descriptor = execute_job({"job": job, "inputs": inputs}, staging, lambda *_: None)
    assert [o["rel_path"] for o in descriptor["outputs"]] == [p for p, _ in RUN_OUTPUTS]
    direct = run_strategy_replay(source, [artifacts.path_of(c.rel_path) for c in caches], "base", tmp_path / "direct")
    replayed = json.loads((staging / "run" / "summary.json").read_text())
    assert replayed["net_pnl"] == direct["net_pnl"]
    column = artifacts.path_of(caches[0].rel_path) / "book.ask_px.npy"
    os.chmod(column, 0o644)  # registered copies are read-only: tampering means overriding that first
    column.write_bytes(b"tampered")
    other = tmp_path / "attempt2"
    other.mkdir()
    with pytest.raises(InvalidDataError):
        execute_job({"job": job, "inputs": inputs}, other, lambda *_: None)


def test_result_from_summary_keeps_exact_clocks_and_decimal_fees() -> None:
    summary = {
        "scenario": "base",
        "valid": False,
        "reason": "not_flat",
        "net_pnl": None,
        "round_trips": 1,
        "fills": 2,
        "orders": 3,
        "decisions": 4,
        "first_event_ns": 2**62 + 1,
        "last_event_ns": 2**62 + 9,
        "fees": {"venue_a/ALPHA-USD": "0.00005"},
    }
    result = result_from_summary(summary)
    assert result["first_event_ns"] == 2**62 + 1 and result["account_valid"] is False and result["net_pnl"] is None
    assert result["fees"] == {"venue_a/ALPHA-USD": "0.00005"}
